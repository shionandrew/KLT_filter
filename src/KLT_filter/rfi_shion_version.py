# Shion's modifications, adapted from Juan's code

"""tools for rfi analysis of baseband data"""

import numpy as np
import time
import datetime
import re
import glob
import os
import scipy.linalg as la
import scipy.constants
import baseband_analysis.core as bbcore
from ch_util import ephemeris, tools
from typing import List, Optional
import baseband_analysis.core.calibration as cal
from baseband_analysis.analysis.beamform import fringestop_time_vectorized
from KLT_filter.injections.simulate_signal import fringestop_delays_vectorized
import ch_util
from beam_model.utils import get_equatorial_from_position, get_position_from_equatorial
import copy
from scipy.linalg import cho_factor, cho_solve
from caput import memh5

def get_good_inputs(
    gains:np.ndarray, #shape (ninputs)
    reordered_inputs,
    ):
    assert np.nanmax(np.abs(gains.flatten()))>0, "all inputs have a gain of 0.0!"
    good_inputs_index = np.where(abs(gains)>0.)[0]
    good_inputs_index_x = np.array([good_inputs_index[ii] for ii in
                                    range(len(good_inputs_index)) if
                                    reordered_inputs[good_inputs_index[ii]].pol=="S"])
    good_inputs_index_y = np.array([good_inputs_index[ii] for ii in
                                    range(len(good_inputs_index)) if
                                    reordered_inputs[good_inputs_index[ii]].pol=="E"])
    good_inputs_index = np.array([good_inputs_index_x, good_inputs_index_y])
    return good_inputs_index #[good_inputs_x,good_inputs_y]


def calculate_fringestop_delays(
    data,
    freq_index,
    ra,
    dec,
    reordered_inputs,
    fringestop_delays=None,
):
    if fringestop_delays is None:
        from baseband_analysis.analysis.beamform import fringestop_time_vectorized
        time = (
        data["time0"]["ctime"][freq_index] + data["time0"]["ctime_offset"][freq_index]
        )
        time = time + np.mean(data.index_map["time"]["offset_s"])

        ra_from_src = np.empty(ra.shape[0])
        dec_from_src = np.empty(dec.shape[0])
        for iipoint in range(len(ra)):
            src = ephemeris.skyfield_star_from_ra_dec(ra[iipoint], dec[iipoint])
            (
                ra_from_src[iipoint],
                dec_from_src[iipoint],
            ) = ephemeris.object_coords(src, time, obs=obs)
        fringestop_phase = fringestop_time_vectorized(
            time,
            data.freq[freq_index],
            reordered_inputs,
            ra_from_src,
            dec_from_src,
            prod_map=prod_map,
            obs=obs,
            static_delays=static_delays,
        ).T.astype(np.complex64)[0]
        time = (data["time0"]["ctime"][freq_index] + data["time0"]["ctime_offset"][freq_index])
        time = time + np.mean(data.index_map["time"]["offset_s"]) #calculate delay at center of the dump
    else:
        fringestop_phase=np.exp(2j*np.pi*data.freq[freq_index]*fringestop_delays) #(nfreq,ninputs)
    return fringestop_phase
def beamform_clean_single_channel(
    data,
    ra,
    dec,
    obs,
    freq_index:int,
    gains,
    static_delays,
    reordered_inputs:List,
    clean:bool,
    POL:bool,
    fringestop_delays:Optional[np.ndarray]=None,
    frame_start=None,
    frame_stop=None,
    **pol_kwargs,
):
    if frame_start is None:
        frame_start=0
    if frame_stop is None:
        frame_stop=0
    
    # Get list of bad inputs from  gains
    good_inputs_index=get_good_inputs(gains[freq_index],reordered_inputs=reordered_inputs)
    N_good_inputs = [len(good_inputs_index[pp]) for pp in range(2)]

    fringestop_phase=calculate_fringestop_delays(
        data,freq_index,ra=ra,dec=dec,fringestop_delays=fringestop_delays,reordered_inputs=reordered_inputs)

    fringestop_phase_x=np.array([fringestop_phase[i] for i in good_inputs_index[0]])
    fringestop_phase_y=np.array([fringestop_phase[i] for i in good_inputs_index[1]])
    fringestop_phase=[fringestop_phase_x,fringestop_phase_y] #ignoring bad inputs

    data_to_beamform=copy.deepcopy(data["baseband"][freq_index])
    ###########################################################################
    ############################ form visibilities ############################
    ###########################################################################
    # Find where NaNs at the end of the timestream are. Same for all inputs

    if clean:
        print("STARTING CLEANING")
        data_for_vis=data_to_beamform[:,~np.isnan(data_to_beamform)[-1]] #remove nans
        data_for_vis=[data_for_vis[good_inputs_index[0]],data_for_vis[good_inputs_index[1]]]
        if POL==True:
            clean_pol_channel(
                data,
                data_for_vis=np.concatenate((data_for_vis[0],data_for_vis[1]),axis=0),
                data_to_beamform=data_to_beamform,
                fringestop_phase=fringestop_phase,
                good_inputs_index=good_inputs_index,
                freq_index=freq_index,
                frame_start=frame_start,
                frame_stop=frame_stop,
                **pol_kwargs,
            )
            print(f"successfully pol cleaned and beamformed index {freq_index}")

        else:
            for pp in range(2):
                # Construct filter
                # The filter matrix has the form np.dot(a, b.T)
                fringestop_phase_pp=fringestop_phase[pp]
                good_inputs_index_pp=good_inputs_index[pp]
                data_for_vis_pp=data_for_vis[pp] #ninputs, ntime
                F=form_F_from_vis(data_for_vis_pp,frame_start,frame_stop)

                S=form_S_from_phase(fringestop_phase_pp)

                c, low = cho_factor(F) 
                v = np.conj(cho_solve((c, low), S))[:,-1]
                # fix phase, <v s^*>=<s s^*>
                z=np.nansum(v*np.conj(fringestop_phase_pp),axis=0) 
                v/=z
                # fix amplitude, <(v s^*)^2>=<(s s^*)**2>
                v_dot_s=np.nansum(np.abs(v*np.conj(fringestop_phase_pp))**2,axis=0)
                s_dot_s=np.nansum(np.abs(fringestop_phase_pp*np.conj(fringestop_phase_pp))**2,axis=0)
                norm_factor=s_dot_s/v_dot_s
                v*=np.sqrt(norm_factor)
                data['tiedbeam_baseband'][freq_index, pp, :]=np.nansum(v[:,np.newaxis] * data_to_beamform[good_inputs_index_pp],axis=0)

            print(f"successfully cleaned and beamformed index {freq_index}")
            data['tiedbeam_baseband'].attrs['cleaned']='True'

    else:
        for pp in range(2):
            good_inputs_index_pp=good_inputs_index[pp]
            fringestop_phase_pp=fringestop_phase[pp]
            baseband_data_pp=data_to_beamform[good_inputs_index_pp]
            data['tiedbeam_baseband'][freq_index, pp, :]= fringestop_phase_pp @ baseband_data_pp 
        print(f"successfully beamformed index {freq_index}")


def clean_pol_channel(
    data,
    data_for_vis,
    data_to_beamform,
    fringestop_phase,
    good_inputs_index,
    freq_index,
    frame_start,
    frame_stop,
    **pol_kwargs,
):
    ###########################################################################
    ############################ form visibilities ############################
    ###########################################################################
    F=form_F_from_vis(data_for_vis,frame_start,frame_stop)
    fringestop_phase_pp=np.concatenate((fringestop_phase[0], fringestop_phase[1]), axis=0)
    S=form_S_from_phase(fringestop_phase_pp)

    N_good_inputs=[len(i) for i in good_inputs_index]
    N_x=N_good_inputs[0]
    S=apply_pol_covariances(S=S,N_x=N_x,**pol_kwargs)

    evalues, R = la.eigh(S, F)
    R_dagger = R.T.conj()
    R_dagger_inv = la.inv(R_dagger)
    a=R_dagger_inv[:, -1] #last 
    b=R_dagger[-1] #last row

    all_inputs=np.concatenate((good_inputs_index[0],good_inputs_index[1]))
    data["baseband"][freq_index, all_inputs] = a[:, np.newaxis] * np.sum(
        b[:, np.newaxis] * data["baseband"][freq_index, all_inputs],
        axis=0)[np.newaxis, :] #b=R
    data['S_F'][freq_index]=(np.abs(b@S)/np.abs(b@F))[0]

    data_to_beamform=data["baseband"][freq_index]
    for pp in range(2):
        good_inputs_index_pp=good_inputs_index[pp]
        fringestop_phase_pp=fringestop_phase[pp]
        baseband_data_pp=data_to_beamform[good_inputs_index_pp]
        data['tiedbeam_baseband'][freq_index, pp, :]= fringestop_phase_pp @ baseband_data_pp 

    data['tiedbeam_baseband'].attrs['cleaned']='True'
    data['tiedbeam_baseband'].attrs['pol_cleaned']='True'

    '''
    L=np.zeros((len(evalues),len(evalues)),dtype=np.complex)
    L[-1,-1]=evalues[-2] #
    L[-1,-1]=evalues[-1]

    R_dagger = R.T.conj()
    R_dagger_inv = la.inv(R_dagger)

    s_i_tensor=np.zeros((N_good_inputs[0]+N_good_inputs[1],2),dtype=complex)

    s_i_tensor[:len(good_inputs_index[0]),0]=np.array(
            [fringestop_phase[0]])
    s_i_tensor[len(good_inputs_index[0]):,1]=np.array(
            [fringestop_phase[1]])

    s_i_tensor=s_i_tensor.conj()

    w=np.matmul(L,R_dagger) 
    w_1=R_dagger_inv[:,-1][:,np.newaxis]*w[-1][np.newaxis,:]
    w_2=R_dagger_inv[:,-2][:,np.newaxis]*w[-2][np.newaxis,:]
    w=np.zeros((2,w.shape[0],w.shape[1]),dtype=complex) #(2, 4, 4)

    w[0]=w_1
    w[1]=w_2

    fringestop_tensor=np.zeros((2,len(fringestop_phase_pp)),dtype=complex)
    fringestop_tensor[0][:len(fringestop_phase[0])]=fringestop_phase[0]
    fringestop_tensor[1][len(fringestop_phase[0]):]=fringestop_phase[1] #(2x4)

    V=np.sum(np.matmul(fringestop_tensor,w),axis=1)
    M=np.matmul(V,s_i_tensor) #(2, 2, 2) #( ,alpha)
    N=la.inv(M)

    KLT_filter=np.matmul(N,V)



    true_signals_tensor=np.zeros((N_good_inputs[0]+N_good_inputs[1],2,data_to_beamform.shape[-1]),dtype=complex)
    true_signals_tensor[:len(good_inputs_index[0]),0]=np.array(
            [data_to_beamform[i] for i in ((good_inputs_index[0]))])

    true_signals_tensor[len(good_inputs_index[0]):,1]=np.array(
            [data_to_beamform[i] for i in ((good_inputs_index[1]))])


    true_signals_tensor=np.swapaxes(true_signals_tensor,0,1)
    signal_=np.matmul(KLT_filter,true_signals_tensor)
    signal_x=signal_[0,0]
    signal_y=signal_[1,1]'''

    #data['tiedbeam_baseband'][freq_index, 0, :]= signal_x
    #data['tiedbeam_baseband'][freq_index, 1, :]= signal_y        



def form_F_from_vis(
    data_for_vis, #shape (ninputs, ntime)
    frame_start:int,
    frame_stop:int,
):
    data_for_vis=data_for_vis[:,frame_start:frame_stop]
    NN=len(data_for_vis)
    F=np.tensordot(data_for_vis,
                        data_for_vis.conj(),
                            axes=([-1], [-1]))/ NN
    return F

def form_S_from_phase(
    fringestop_phase:np.ndarray,
):
    signal_phase=fringestop_phase.conj()
    S=(np.conj(signal_phase[np.newaxis, :])*(
            signal_phase[:, np.newaxis]))
    return S

def apply_pol_covariances(
    S:np.ndarray,
    N_x:int,
    Ex_Ex:float,
    Ex_Ey:complex,
    Ey_Ex:complex,
    Ey_Ey:float,
):
    rows=S.shape[0]
    cols=S.shape[1]

    # Create masks for different regions
    upper_left_mask = np.logical_and(np.arange(rows) < N_x, np.arange(cols) < N_x)
    upper_right_mask = np.logical_and(np.arange(rows) < N_x, np.arange(cols) >= N_x)
    lower_left_mask = np.logical_and(np.arange(rows) >= N_x, np.arange(cols) < N_x)
    lower_right_mask = np.logical_and(np.arange(rows) >= N_x, np.arange(cols) >= N_x)

    # Modify S based on masks
    S[upper_left_mask] *= Ex_Ex
    S[upper_right_mask] *= Ex_Ey
    S[lower_left_mask] *= Ey_Ex
    S[lower_right_mask] *= Ey_Ey

    return S