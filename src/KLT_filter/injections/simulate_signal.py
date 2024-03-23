from outriggers_vlbi_pipeline.multibeamform import beamform_multipointings, beamform_calibrators,rebeamform_singlebeam
from outriggers_vlbi_pipeline.vlbi_pipeline_config import kko_backend, chime_backend,frb_events_database,kko_events_database
from outriggers_vlbi_pipeline.query_database import get_event_data,find_files,get_calibrator_dataframe
from outriggers_vlbi_pipeline.cross_correlate_data import correlate_multibeam_data
from outriggers_vlbi_pipeline.query_database import check_correlation_completion, update_event_status,check_baseband_localization_completion
from outriggers_vlbi_pipeline.calibration import update_calibrator_list
from dtcli.src import functions
from dtcli.utilities import cadcclient
from ch_util import tools
import numpy as np
from outriggers_vlbi_pipeline.multibeamform import get_calfiles
from outriggers_vlbi_pipeline.query_database import update_event_status,get_event_data, get_full_filepath, find_files,fetch_data_from_sheet,check_correlation_completion,get_target_vis_files,get_cal_vis_files
from baseband_analysis.core.sampling import fill_waterfall
import ch_util
import scipy
from astropy.coordinates import SkyCoord
import astropy.units as un
import os 
import glob
import datetime
import pandas as pd
import gspread
import time
import subprocess
import logging
import parser
import argparse
import shutil
import traceback
import sys
import re
from pathlib import Path
import time
from outriggers_vlbi_pipeline.query_database import get_full_filepath, get_cal_vis_files
from outriggers_vlbi_pipeline.known_calibrators import get_true_pulsar_pos
from outriggers_vlbi_pipeline.geometry import angular_distance
from outriggers_vlbi_pipeline.arc_commands import datatrail_pull_or_clear,datatrail_pull_cmd,datatrail_clear_cmd,baseband_exists,delete_baseband,vchmod,delete_multibeam,data_exists_at_minoc,datatrail_pull,datatrail_clear
import logging
from outriggers_vlbi_pipeline.vlbi_pipeline_config import (
    chime,
    kko,
    gbo,
    current_version,
    kko_events_database,
)
from outriggers_vlbi_pipeline.vlbi_pipeline_config import calibrator_catalogue, calibrator_database, current_calibrators,current_version,known_pulsars
from outriggers_vlbi_pipeline.cross_correlate_data import re_correlate_target
from pycalc11 import Calc

from outriggers_vlbi_pipeline.calibration import get_calibrator_visibilities, make_calibrated_visibilities
from pyfx.core_correlation_station import get_delays, frac_samp_shift,getitem_zp1d,get_start_times
import astropy.coordinates as ac
from astropy.time import Time, TimeDelta


chime = ac.EarthLocation.from_geocentric(
    x=-2059166.313 * un.m, y=-3621302.972 * un.m, z=4814304.113 * un.m
)
chime.info.name = "chime"

kko = ac.EarthLocation.from_geocentric(
    x = (-2111738.254-13.990) * un.m,
    y = (-3581458.222+4.666) * un.m, 
    z = (4821611.987-1.906) * un.m
)
kko.info.name = 'kko'


class baseband_data_sim:
    def __init__(
        self, 
        telescope: ac.EarthLocation, 
        data, 
        inputs,
        ):
        self.inputs = get_reordered_inputs(data,inputs)
        self.telescope = telescope
        x_indices,y_indices = get_pol_indices(data,inputs)
        self.x_indices=x_indices
        self.y_indices=y_indices

def get_reordered_inputs(data,inputs):
    # Get gains (nfreq,ninputs) (ordered to bbdata)

    # order feed positions (ordered to bbdata)
    converted_data_inputs = data.input.astype(
            [("chan_id", "<u2"), ("correlator_input", "U32")]
        )
    reordered_inputs = tools.reorder_correlator_inputs(
                converted_data_inputs, inputs
            )
    return reordered_inputs
    
def get_pol_indices(data,inputs):
    pols=[]
    reordered_inputs=get_reordered_inputs(data=data,inputs=inputs)
    for i in reordered_inputs:
        try:
            pols.append(i.pol)
        except Exception as e:
            pols.append('')
    pols=np.array(pols)#[i.pol for i in reordered_inputs])
    x=np.where(pols=='E')[0]
    y=np.where(pols=='S')[0]
    return x,y


def inject_signal_into_baseband(
    tel_bbdata,
    signal, #2d array
    corr_inputs,
    delays,
    s_t_n,
    frame_start,
    frame_stop,
    tec=0,
    sample_rate=2.56,#microseconds
    inter_channel_sign=-1, #since baseband data is conjugated w.r.t. sky
    ):
    freqs=tel_bbdata.freq
    delayed_signals=[]
    for pp in range(2):
        delayed_signal=generate_time_shifted_signal(
            signal=signal[pp],
            delays=delays,
            freqs=freqs,
            sample_rate=sample_rate,
            inter_channel_sign=inter_channel_sign
        )
        # add ionospheric delay
        k_dm = 1344.54095924  # Mhz/Tecu
        ionophase = k_dm*tec/freqs
        P = np.exp(2j * np.pi * (ionophase))  # nfreq
        delayed_signal*=P[:,np.newaxis,np.newaxis]

        delayed_signals.append(delayed_signal)
    

    for freq_index in range(len(freqs)):
        x_inputs,y_inputs=get_pol_indices(data=tel_bbdata,inputs=corr_inputs)
        inputs=[x_inputs,y_inputs]
        for pp in range(len(inputs)):
            input_indices=inputs[pp]
            delayed_signal_stn=delayed_signals[pp][freq_index][input_indices]#ninputs, ntime
            #noise=np.nanstd(np.nansum(np.abs(tel_bbdata['baseband'][freq_index,input_indices])**2,axis=0)[frame_start:frame_stop],axis=-1) #shape ntime, std over ntime after summing inputs 
            #delayed_signal_stn=delayed_signal_stn*np.sqrt(noise)*s_t_n  #amplitude of signal determined by s_t_n
            bbdata_ntime=tel_bbdata['baseband'][freq_index].shape[-1]
            sig_ntime=delayed_signal_stn.shape[-1]
            if bbdata_ntime<sig_ntime:
                tel_bbdata['baseband'][freq_index][input_indices]+=delayed_signal_stn[:,:bbdata_ntime]

            elif bbdata_ntime>sig_ntime:
                tel_bbdata['baseband'][freq_index][input_indices][:,:sig_ntime]+=delayed_signal_stn

            else:
                tel_bbdata['baseband'][freq_index][input_indices]+=delayed_signal_stn
        tel_bbdata.attrs['tec']=tec



def generate_time_shifted_signal(
    signal,
    delays, #microseconds
    freqs:np.ndarray, #Mhz
    sample_rate=2.56,#microseconds
    inter_channel_sign=-1, #since baseband data is conjugated w.r.t. sky
    beamformed:bool=False,
    ):
    '''
    if beamformed (baseband), returns delayed signal array for each input of size n_input x n_frames  
    if not beamformed (tiedbeam_baseband), returns delayed signal array for each channel n_frames  
    '''
    int_delay=int(np.mean(np.round(delays/sample_rate)))
    
    subint_delays=delays-int_delay*sample_rate
    if int_delay!=0:
        signal_delayed=np.roll(signal,int_delay,axis=-1)
    else:
        signal_delayed=signal

    if not beamformed:
        assert max(subint_delays)<sample_rate, "integer delay is not constant across inputs"
        subint_phase=np.exp(2j*np.pi*freqs[:,np.newaxis]*subint_delays[np.newaxis,:]*inter_channel_sign) #(nfreq,ninputs)
        
        signal_delayed=signal_delayed[np.newaxis,np.newaxis,:]*subint_phase[...,np.newaxis]

    else:
        subint_phase=np.exp(2j*np.pi*freqs*subint_delays*inter_channel_sign) #shape (nfreq)        
        signal_delayed=signal_delayed*subint_phase[:,np.newaxis,np.newaxis]

    return signal_delayed

    
def get_input_delays(
    feeds,
    station,
    ctime,
    ra,
    dec,
    static_delays):#,inter_channel_sign=-1):
    antenna_delays=fringestop_delays_vectorized(ctime,feeds,np.array([ra]),np.array([dec]),reference_feed=-1,obs=ch_util.ephemeris.chime,static_delays=static_delays)[:,0]
    baseline_delay=0
    if station.info.name != chime.info.name:
        # need additional delay between outrigger + chime 
        telescopes=[chime,station]
        start_time=Time(
                    ctime,
                    format="unix",
                    precision=9,
                )
        srcs = ac.SkyCoord(
            ra=np.array([ra]),
            dec=np.array([dec]),
            unit='deg',
            frame='icrs',
        )
        tel_names=[str(tel.info.name for tel in telescopes)]
    
        ci = Calc(
        station_names=tel_names,
        station_coords=telescopes, #array of geocentric positions, 1st is fringestopping center, rest are 
        source_coords=srcs,
        start_time=start_time, #10s buffer
        duration_min=1,
        base_mode='geocenter', 
        dry_atm=True, 
        wet_atm=True,
        d_interval=1,
    )
        ci.run_driver()    
        baseline_delay=get_delays(1,ref_start_time=np.array([ctime]),ref_start_time_offset=np.array([0]),
                    wij=1,pycalc_results=ci,ref_frame=0)[0][0]  #nfreq,nframe

    return antenna_delays,baseline_delay

            


# taken straight from baseband_analysis, modified to return delays 
def fringestop_delays_vectorized(
    time,
    feeds,
    src_ra,
    src_dec,
    reference_feed=-1,
    obs=ch_util.ephemeris.chime,
    static_delays=True,
):
    """Rewrittern ch_util.tools.fringestop_time optimized for tied_array.

    Parameters
    ----------
    time : double
        The time.
    feeds : list of CorrInputs
        The feeds in the timestream.
    src_ra : list
        List of ra values from ch_util.ephemeris.object_coords.
        Should have length # of beams.
    src_dec : list
        List of dec values from ch_util.ephemeris.object_coords.
        Should have length # of beams.
    obs : `caput.time.Observer`
        An observer instance to use. If not supplied use `chime`. For many
        calculations changing from this default will make little difference.

    Returns
    -------
    fringestopped_timestream : np.ndarray[nprod, npointings]
        Fringestop phase array that will need to be transposed and masked later.
        Type is np.complex128.
    """
    ra = obs.unix_to_lsa(time)
    ha = np.radians(ra) - src_ra  # 1d array with src_ra varying
    latitude = np.radians(obs.latitude)
    # Get feed positions / c
    feedpos = (
        ch_util.tools.get_feed_positions(feeds, get_zpos=False) / scipy.constants.c
    )
    feed_delays = np.array([f.delay for f in feeds])

    # https://github.com/radiocosmology/caput/blob/master/caput/interferometry.py
    x, y = feedpos.T[..., np.newaxis]
    delay_ref = np.tile(x, (1, src_dec.shape[0])) * (-1 * np.cos(src_dec) * np.sin(ha))
    delay_ref += np.tile(y, (1, src_dec.shape[0])) * (
        np.cos(latitude) * np.sin(src_dec)
        - np.sin(latitude) * np.cos(src_dec) * np.cos(ha)
    )

    # add in the static delays
    if static_delays:
        delay_ref += feed_delays[:, np.newaxis]

    # Calculate baseline separations and pack into product array
    delays = delay_ref - delay_ref[reference_feed]

    # then vectorizing fringestop_array
    # Set any non CHIME feeds to have zero phase
    delays = np.nan_to_num(delays, copy=False) * 1e6 #us

    return delays[:-1] #exclude reference antenna


    
def polar_prob(w,y,I=1,p=0.9,Q=0,U=0.35):
    assert p**2*I-U**2-Q**2>0, "please enter valid polarization parameters"
    V=np.sqrt(p**2*I-U**2-Q**2)
    Ex_Ex=(I+Q)/2
    Ey_Ey=(I-Q)/2
    Ex_Ey=(U+1j*V)/2
    Ey_Ex=(U-1j*V)/2

    Gamma = np.array([[Ex_Ex, Ex_Ey], [Ey_Ex, Ey_Ey]])# diagonal covariance
    prefactor=1/np.pi**2*np.sqrt(np.linalg.det(Gamma)*np.linalg.det(np.conj(Gamma)))
    assert np.imag(prefactor)==0.0, print((prefactor))
    prefactor=np.abs(prefactor)
    Big_Gamma=[
        [Ex_Ex, Ex_Ey,0,0],
        [Ey_Ex, Ey_Ey,0,0,],
        [0,0,np.conj(Ex_Ex),np.conj(Ex_Ey)],
        [0,0,np.conj(Ey_Ex),np.conj(Ey_Ey)]]
    Big_Gamma=np.array(Big_Gamma)
    Big_Gamma_inv=np.linalg.inv(Big_Gamma)
    
    Z_1=np.zeros((1,4),np.complex)
    Z_2=np.zeros((4,1),np.complex)

    
    Z_1[0,0]=np.conj(w)
    Z_1[0,1]=np.conj(y)
    Z_1[0,2]=w
    Z_1[0,3]=y

    Z_2[0,0]=w
    Z_2[1,0]=y
    Z_2[2,0]=np.conj(w)
    Z_2[3,0]=np.conj(y)
    
    exp_factor=-0.5*np.matmul(Z_1,np.matmul(Big_Gamma_inv,Z_2))
    assert np.abs(np.imag(exp_factor))< 0.00001
    exp_factor = np.real(exp_factor)
    assert exp_factor.shape==(1,1)
    exponent=np.exp(exp_factor[0,0])
    assert np.abs(np.imag(exponent))< 0.00001
    exponent=np.abs(exponent)
    return prefactor*exponent
def uniform_proposal(x, delta=0.250):
    a=np.random.uniform(np.real(x[0]) - delta, np.real(x[0]) + delta)
    b=np.random.uniform(np.imag(x[0]) - delta, np.imag(x[0]) + delta)
    c=np.random.uniform(np.real(x[1]) - delta, np.real(x[1]) + delta)
    d=np.random.uniform(np.imag(x[1]) - delta, np.imag(x[1]) + delta)
    return np.array([a+b*1j,c+d*1j])

def metropolis_sampler(p, nsamples, proposal=uniform_proposal):
    x = np.array([0+0.0j,0+0.0j]) # start somewhere

    for i in range(nsamples):
        trial = proposal(x) # random neighbour from the proposal distribution
        acceptance = p(trial)/p(x)

        # accept the move conditionally
        if np.random.uniform() < acceptance:
            x = trial

        yield x
        
def generate_gaussian_signal(
    data,
    frame_start=100,
    frame_stop=1000):
    out_signal=np.zeros((2,data.ntime),dtype=data['baseband'].dtype)
    t_vals=np.arange(frame_start,frame_stop)
    center=(frame_stop+frame_start)//2
    width=(frame_start-frame_stop)//2
    N=len(t_vals)
        
    p = lambda x: polar_prob(x[0],x[1])
    samples = np.array(list(metropolis_sampler(p, N)))
    out_signal[0,frame_start:frame_stop]+=samples[:,0]
    out_signal[1,frame_start:frame_stop]+=samples[:,1]

    return out_signal