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
from clean_correlation import clean_and_form_singlebeams
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


def inject_pulse(
    data,
    signal, #signal at reference station
    good_inputs,
    telescope,
    inputs,
    ra,
    dec,
    freq_index,
    s_t_n,
    mean_delays,
    ):
    ''' add pulse to bbdata object '''
    # fringestop outriggers-> chime
    inputs_to_use=[inputs[i] for i in good_inputs]
    feed_positions=convert_inputs_to_stations(inputs_to_use,telescope) #inputs should be ordered to correspond to baseband data
    phases_all=[]
    freq=data.index_map['freq']['centre'][freq_index]
    ctime=data['time0']['ctime'][freq_index]
    mean_delays_to_use=[mean_delays[i] for i in good_inputs]
    S,phases=construct_S(feed_positions,ctime,ra,dec,freq,mean_delays=mean_delays)
    
    injection=generate_time_shifted_signal(
        data=data,signal=signal,good_inputs=good_inputs,phases=phases,freq_index=freq_index,
        s_t_n=s_t_n)
    
    data['baseband'][freq_index][good_inputs]+=injection

    return data,S,phases


def convert_inputs_to_stations(inputs,telescope):
    antennas=[]
    ''' The first column is the E-W position (x)
        (increasing to the E), and the second is the N-S positio (y) (increasing
        to the N). Non CHIME feeds get set to `NaN`.'''
    # ONLY give inputs with actual positions
    for n_ant in range(len(inputs)):
        input=inputs[n_ant]
        x=input.pos[0]* un.m+telescope.x#* un.m
        y=input.pos[1]* un.m+telescope.y#* un.m
        z=input.pos[2]* un.m+telescope.z#* un.m
        antenna=ac.EarthLocation.from_geocentric(x=x, y=y,z=z)
        antenna.info.name=f'{telescope.info.name}_antenna_'+str(n_ant)
        antennas.append(antenna)
    return antennas


def generate_time_shifted_signal(data,signal,good_inputs,phases,freq_index,s_t_n):
    # data: np.ndarray of shape (freq,inputs,time) (should be single frequency)
    noise=np.nanstd(np.abs(data['baseband'][freq_index,good_inputs])**2,axis=-1) #shape ninputs,ntime, average along ntime for each input 
    phases_amplitude=phases*np.sqrt(noise)*s_t_n 
    time_shifted_injection=phases_amplitude[:,np.newaxis]*signal[np.newaxis,:]  #of length ninputs,ntime    
    return time_shifted_injection
    
    
def get_input_delays(feeds,station,ctime,ra,dec,static_delays):#,inter_channel_sign=-1):
    antenna_delays=fringestop_delays_vectorized(ctime,feeds,ra,dec,reference_feed=-1,obs=ch_util.ephemeris.chime,static_delays=static_delays)
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
        antenna_delays+=baseline_delay

    return antenna_delays

            
def construct_S(feed_positions,ctime,ra,dec,freq):#,inter_channel_sign=-1):
    """ 
    inter_channel_sign: complex conjugated phases
    for agiven frequency
    we assume that the delay changes microscopically over the course of a scan. 
     We also choose to ignore the frame delay for now... """
    S_matrix=np.zeros((len(mean_delays),len(mean_delays)),dtype=np.complex64)
    phases=[]
    for antenna_a in range(len(mean_delays)):
        phases.append(np.exp(2j*np.pi*(freq*mean_delays[antenna_a])))
        for antenna_b in range(len(mean_delays)):
            relative_delay=mean_delays[antenna_a]-mean_delays[antenna_b]
            S_matrix[antenna_a,antenna_b]=np.exp(2j*np.pi*(freq*relative_delay)) #we just ignore the 2pi phase ambiguity for now...
    return S_matrix,np.array(phases)




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

    #fs_timestream = 2.0j * np.pi * delays * freq * 1e6
    #fs_timestream = np.exp(fs_timestream, out=fs_timestream)
    #return fs_timestream
    return delays