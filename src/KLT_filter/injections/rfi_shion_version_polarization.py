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
def beamform_clean_rfi(
    data,
    ra:np.ndarray, 
    dec:np.ndarray, 
    source_names:np.ndarray,
    ps_gain_file:str, 
    inputs:List,
    static_delays:bool,
    clean:bool,
    Ex_Ey:Optional[np.complex128]=None,
    Ey_Ex:Optional[np.complex128]=None, 
    Ex_Ex:Optional[np.complex128]=None, 
    Ey_Ey:Optional[np.complex128]=None, 
    fringestop_delays:Optional[np.ndarray]=None, #freq indep
    frame_start=None,
    frame_stop=None,
    reference_feed=tools.CHIMEAntenna(
        id=-1, slot=-1, powered=True, flag=True, pos=[0, 0, 0]
    ),
    obs=ephemeris.chime,
    delayed_signals=None,
    ):
    """
    Clean baseband data from persistent RFI
    
    Parameters:
    -----------
    data: core.BBdata
        baseband data
    ps_gain_file: str
        Point source gain file name (full path)
    rfi_channels: array
        List of persistent RFI channel ids to be cleaned
    verbose: bool
        Verbose
    """

    # baseband data info
    N_freqs = len(data.freq)

    # Get gains (nfreq,ninputs) (ordered to bbdata)
    freq_id_list = data.index_map["freq"]["id"]
    gain = cal.read_gains(ps_gain_file)[freq_id_list] #only get relevent frequencies
    gain_reordered = gain[:, data.input["chan_id"]] # reordered gains by channel input

    # order feed positions (ordered to bbdata)
    converted_data_inputs = data.input.astype(
            [("chan_id", "<u2"), ("correlator_input", "U32")]
        )
    reordered_inputs = tools.reorder_correlator_inputs(
                converted_data_inputs, inputs
            )

    prod_map = np.empty(len(data.input), dtype=[("input_a", "u2"), ("input_b", "u2")])
    prod_map["input_a"] = np.arange(len(data.input))

    reordered_inputs.append(reference_feed)
    prod_map["input_b"] = len(data.input)

    nfreq=data.baseband.shape[0]
    npointing=1
    npol=2
    ntime=data.baseband.shape[-1]
    tiedbeam_baseband = data.create_dataset(
        "tiedbeam_baseband", shape=(nfreq, npointing*npol, ntime), dtype=data.baseband.dtype
    )
    from caput import memh5
    memh5.copyattrs(data["baseband"].attrs, tiedbeam_baseband.attrs)
    tiedbeam_baseband.attrs["axis"] = ["freq", "beam", "time"]
    tiedbeam_baseband.attrs["conjugate_beamform"] = int(1)


    for rfi_ind in range(N_freqs):
        if True:#try:
            clean_single_channel(
            data=data,
            ra=ra,
            dec=dec,
            obs=obs,
            Ex_Ex=Ex_Ex,
            Ey_Ey=Ey_Ey,
            Ex_Ey=Ex_Ey,
            Ey_Ex=Ey_Ex,
            gain_reordered=gain_reordered,
            fringestop_delays=fringestop_delays,
            rfi_ind=rfi_ind,
            prod_map=prod_map,
            reordered_inputs=reordered_inputs,
            static_delays=static_delays,
            clean=clean,
            frame_start=frame_start,
            frame_stop=frame_stop,
            delayed_signals=delayed_signals,
        )
            print(f"successfully cleaned channel {data.freq[rfi_ind]}")

        else:#except Exception as e:
            ct=datetime.datetime.now()
            print(f"{ct.strftime('%Y%m%dT%H%M%S')} WARNING: COULD NOT PROCESS CHANNEL {data.freq[rfi_ind]} due to error: {e}")
    ct=datetime.datetime.now()
    print("%s: RFI cleaning finished" %(ct.strftime("%Y%m%dT%H%M%S")))
    
    del data['baseband']
    ib_dtype = [
        ("ra", float),
        ("dec", float),
        ("x_400MHz", float),
        ("y_400MHz", float),
        ("pol", "S1"),
    ]
    ra, dec = (
        np.asarray(ra, dtype=float).reshape(-1),
        np.asarray(dec, dtype=float).reshape(-1),
    )
    if source_names is not None:
        ib_dtype.append(("source_name", "<S50"))
    ib = np.empty(npointing*npol, dtype=ib_dtype)
    ib["ra"] = ra#(ra[:, None] * np.ones(2, dtype=ra.dtype)).flat
    ib["dec"] = dec#(dec[:, None] * np.ones(2, dtype=dec.dtype)).flat
    ctime = data["time0"]["ctime"][-1] + data["time0"]["ctime_offset"][-1]
    ctime = ctime + np.mean(data.index_map["time"]["offset_s"])
    x, y = get_position_from_equatorial(ra, dec, ctime, telescope_rotation_angle=None)
    ib["x_400MHz"] = x#(x[:, None] * np.ones(2, dtype=x.dtype)).flat
    ib["y_400MHz"] = y#(y[:, None] * np.ones(2, dtype=y.dtype)).flat
    ib["pol"] = ["S", "E"] * npointing
    if source_names is not None:
        ib["source_name"] = [y for x in source_names for y in (x,) * 2]
    loc = data.create_dataset("tiedbeam_locations", data=ib)
    loc.attrs["axis"] = ["beam"]

    return

def clean_single_channel(
    data,
    ra,
    dec,
    obs,
    Ex_Ey:np.complex128,
    Ey_Ex:np.complex128, 
    Ex_Ex: np.complex128, 
    Ey_Ey: np.complex128, 
    rfi_ind:int,
    gain_reordered,
    prod_map,
    static_delays,
    reordered_inputs:List,
    clean:bool,
    fringestop_delays:Optional[np.ndarray]=None,
    frame_start=None,
    frame_stop=None,
    delayed_signals=None,
):

    if frame_start is None:
        frame_start=0
    if frame_stop is None:
        frame_stop=0
    
    time = (data["time0"]["ctime"][rfi_ind] + data["time0"]["ctime_offset"][rfi_ind])
    time = time + np.mean(data.index_map["time"]["offset_s"])
    # Get list of bad inputs from  gains
    assert np.nanmax(np.abs(gain_reordered[rfi_ind].flatten()))>0, "all inputs have a gain of 0.0; gains appear to have zeroed out for this channel!"
    good_inputs_index = np.where(abs(gain_reordered[rfi_ind])>0.)[0]
    good_inputs_index_x = np.array([good_inputs_index[ii] for ii in
                                    range(len(good_inputs_index)) if
                                    reordered_inputs[good_inputs_index[ii]].pol=="S"])
    good_inputs_index_y = np.array([good_inputs_index[ii] for ii in
                                    range(len(good_inputs_index)) if
                                    reordered_inputs[good_inputs_index[ii]].pol=="E"])
    good_inputs_index = np.array([good_inputs_index_x, good_inputs_index_y])
    good_inputs_index_all=np.concatenate((good_inputs_index_x, good_inputs_index_y), axis=0)


    N_good_inputs = [len(good_inputs_index[pp]) for pp in range(2)]
    if fringestop_delays is None:
        fringestop_delays=fringestop_delays_vectorized(
            time,
            feeds=reordered_inputs,
            src_ra=ra,
            src_dec=dec,
            reference_feed=-1,
            obs=obs,
            static_delays=static_delays)[:,0]

    fringestop_phase=np.exp(2j*np.pi*data.freq[rfi_ind]*fringestop_delays) #(nfreq,ninputs)
    baseband_data_unclean=copy.deepcopy(data.baseband[:])

    ###########################################################################
    ############################ form visibilities ############################
    ###########################################################################
    # Find where NaNs at the end of the timestream are. Same for all inputs

    fringestop_phasepp=np.array([fringestop_phase[i] for i in good_inputs_index_all])
    baseband_data=baseband_data_unclean[rfi_ind][good_inputs_index_all] #ninputs, ntime
    baseband_data=baseband_data[...,frame_stop:-frame_stop//2] #remove signal when constructing visibility matrix
    baseband_data=baseband_data[:,~np.isnan(baseband_data)[-1]] #remove nans

    real_data_xx=baseband_data_unclean[rfi_ind, good_inputs_index_x]#baseband_data_unclean[rfi_ind, good_inputs_index_x]
    real_data_yy=baseband_data_unclean[rfi_ind, good_inputs_index_y]#baseband_data_unclean[rfi_ind, good_inputs_index_y]
    real_data=np.concatenate((real_data_xx,real_data_yy),axis=0)

    NN=len(baseband_data)
    vis=(mat2utvec(np.tensordot(baseband_data,
                                baseband_data.conj(),
                                axes=([-1], [-1]))) / NN)

    print("STARTING CLEANING")
    fringestop_phase_x=fringestop_phasepp[:len(good_inputs_index_x)]
    fringestop_phase_y=fringestop_phasepp[len(good_inputs_index_x):]

    FRINGESTOP=True
    if False:#delayed_signals is not None:
        window=(frame_stop-frame_start)//4
        delayed_signals_x=delayed_signals[0,rfi_ind,good_inputs_index_x,frame_start+window:frame_stop-window]
        #signal_x=np.mean(delayed_signals_x,axis=0)
        delayed_signals_y=delayed_signals[1,rfi_ind,good_inputs_index_y,frame_start+window:frame_stop-window]
        #signal_y=np.mean(delayed_signals_y,axis=0)
        if FRINGESTOP:
            delayed_signals_x*=fringestop_phase_x[:,np.newaxis]
            delayed_signals_y*=fringestop_phase_y[:,np.newaxis]

        true_baseband=np.concatenate((delayed_signals_x,delayed_signals_y),axis=0)
        vis_signal=(mat2utvec(np.tensordot(true_baseband,
                                    true_baseband.conj(),
                                    axes=([-1], [-1]))) / NN)
        S_signal= utvec2mat(N_good_inputs[0]+N_good_inputs[1], vis_signal)#fs_phase.conj())

    if FRINGESTOP:
        real_data_xx=real_data_xx*fringestop_phase_x[:,np.newaxis]
        real_data_yy=real_data_yy*fringestop_phase_y[:,np.newaxis]
        real_data=np.concatenate((real_data_xx,real_data_yy),axis=0)

        baseband_data*=fringestop_phasepp[:,np.newaxis]
        vis=(mat2utvec(np.tensordot(baseband_data,
                                    baseband_data.conj(),
                                    axes=([-1], [-1]))) / NN)
        fringestop_phasepp=fringestop_phasepp*0+1
        fringestop_phase_x=fringestop_phase_x*0+1
        fringestop_phase_y=fringestop_phase_y*0+1
        fringestop_phase=fringestop_phase*0+1

    
    fs_phase=(np.conj(fringestop_phasepp[np.newaxis, :])*(
                fringestop_phasepp[:, np.newaxis]))

    fs_phase=mat2utvec(fs_phase)
    
    S = utvec2mat(N_good_inputs[0]+N_good_inputs[1], fs_phase.conj())
    F = utvec2mat(N_good_inputs[0]+N_good_inputs[1], vis) #should signal be extracted first?

    rows=S.shape[0]
    cols=S.shape[1]
    N_x=N_good_inputs[0]
    TEST=False
    if TEST:
        Ex_Ey=0
        Ey_Ex=0
        Ex_Ex=1
        Ey_Ey=1
    TEST2=TEST
    for row in range(rows):
        for col in range(cols):
            if row<N_x:
                if col>=N_x:
                    S[row,col]*=Ex_Ey
                    if TEST2:
                        F[row,col]=0*1j
                else:
                    S[row,col]*=Ex_Ex

            else:  
                if col<N_x:
                    S[row,col]*=Ey_Ex
                    if TEST2:
                        F[row,col]=0*1j
                else:
                    S[row,col]*=Ey_Ey

    ### REMOVE ###
    #S=S_signal
    ### REMOVE ###
    F=F-S
    evalues, R = la.eigh(S, F)


    L=np.zeros((len(evalues),len(evalues)),dtype=np.complex)
    L[-2,-2]=evalues[-2]
    L[-1,-1]=evalues[-1]

    R_dagger = R.T.conj()
    R_dagger_inv = la.inv(R_dagger)

    s_i_tensor=np.zeros((N_good_inputs[0]+N_good_inputs[1],2),dtype=complex)

    s_i_tensor[:len(good_inputs_index[0]),0]=np.array(
            [fringestop_phase[i] for i in good_inputs_index[0]])
    s_i_tensor[len(good_inputs_index[0]):,1]=np.array(
            [fringestop_phase[i] for i in good_inputs_index[1]])

    s_i_tensor=s_i_tensor.conj()

    w=np.matmul(L,R_dagger) 
    w_1=R_dagger_inv[:,-1][:,np.newaxis]*w[-1][np.newaxis,:]
    w_2=R_dagger_inv[:,-2][:,np.newaxis]*w[-2][np.newaxis,:]
    w=np.zeros((2,w.shape[0],w.shape[1]),dtype=complex) #(2, 4, 4)

    w[0]=w_1
    w[1]=w_2

    fringestop_tensor=np.zeros((2,len(fringestop_phasepp)),dtype=complex)
    fringestop_tensor[0][:len(fringestop_phase_x)]=fringestop_phase_x
    fringestop_tensor[1][len(fringestop_phase_x):]=fringestop_phase_y #(2x4)

    V=np.sum(np.matmul(fringestop_tensor,w),axis=1)
    M=np.matmul(V,s_i_tensor) #(2, 2, 2) #( ,alpha)
    N=la.inv(M)

    KLT_filter=np.matmul(N,V)
    true_signals_tensor=np.zeros((N_good_inputs[0]+N_good_inputs[1],2,real_data.shape[-1]),dtype=complex)
    true_signals_tensor[:len(good_inputs_index[0]),0]=np.array(
            [real_data[i] for i in range(len(good_inputs_index[0]))])

    true_signals_tensor[len(good_inputs_index[0]):,1]=np.array(
            [real_data[i] for i in range(len(good_inputs_index[0]),N_good_inputs[0]+N_good_inputs[1])])


    true_signals_tensor=np.swapaxes(true_signals_tensor,0,1)
    signal_=np.matmul(KLT_filter,true_signals_tensor)
    signal_x=signal_[0,0]
    signal_y=signal_[1,1]

    
    data['tiedbeam_baseband'][rfi_ind, 0, :]= signal_x
    data['tiedbeam_baseband'][rfi_ind, 1, :]= signal_y        
    data['tiedbeam_baseband'].attrs['cleaned']='True'
    data['tiedbeam_baseband'].attrs['ExEx']=Ex_Ex
    data['tiedbeam_baseband'].attrs['ExEy']=Ex_Ey
    data['tiedbeam_baseband'].attrs['EyEx']=Ey_Ex
    data['tiedbeam_baseband'].attrs['EyEy']=Ey_Ey


def mat2utvec(A):
    """Vectorizes its upper triangle of the (hermitian) matrix A.

    Parameters
    ----------
    A : 2d array
        Hermitian matrix

    Returns
    -------
    1d array with vectorized form of upper triangle of A

    Example
    -------
    if A is a 3x3 matrix then the output vector is
    outvector = [A00, A01, A02, A11, A12, A22]

    See also
    --------
    utvec2mat
    """

    iu = np.triu_indices(np.size(A, 0)) # Indices for upper triangle of A

    return A[iu]


def utvec2mat(n, utvec):
    """Recovers a hermitian matrix a from its upper triangle vectorized version.

     Parameters
     ----------
     n : int
         order of the output hermitian matrix
     utvec : 1d array
         vectorized form of upper triangle of output matrix

    Returns
    -------
    A : 2d array
        hermitian matrix
    """

    iu = np.triu_indices(n)
    A = np.zeros((n, n), dtype=utvec.dtype)
    A[iu] = utvec # Filling uppper triangle of A
    A = A+np.triu(A, 1).conj().T # Filling lower triangle of A
    return A
