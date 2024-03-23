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

def clean_persistent_rfi(
    data,
    ra:np.ndarray, 
    dec:np.ndarray, 
    Ex_Ey:np.complex128,
    Ey_Ex:np.complex128, 
    Ex_Ex: np.complex128, 
    Ey_Ey: np.complex128, 
    ps_gain_file:str, 
    inputs:List,
    static_delays:bool,
    clean:bool,
    fringestop_delays:Optional[np.ndarray]=None, #freq indep
    frame_start=None,
    frame_stop=None,
    reference_feed=tools.CHIMEAntenna(
        id=-1, slot=-1, powered=True, flag=True, pos=[0, 0, 0]
    ),
    obs=ephemeris.chime):
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
    nbeam=1
    npol=2
    ntime=data.baseband.shape[-1]
    tiedbeam_baseband = data.create_dataset(
        "tiedbeam_baseband", shape=(nfreq, nbeam*npol, ntime), dtype=data.baseband.dtype
    )
    from caput import memh5
    memh5.copyattrs(data["baseband"].attrs, tiedbeam_baseband.attrs)
    tiedbeam_baseband.attrs["axis"] = ["freq", "beam", "time"]


    for rfi_ind in range(N_freqs):
        try:
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
            frame_stop=frame_stop
        )
            print(f"successfully cleaned channel {data.freq[rfi_ind]}")

        except Exception as e:
            print(f"WARNING: COULD NOT PROCESS CHANNEL {data.freq[rfi_ind]} due to error: {e}")

    print("%s: RFI cleaning finished" %(time.strftime("%Y%m%dT%H%M%S")))
    
    del data['baseband']

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
):
    if frame_start is None:
        frame_start=min(np.where(signal==0.0)[0].flatten()) #for constructing the noise matrix
        frame_stop=max(np.where(signal==0.0)[0].flatten())
    
    time = data["time0"]["ctime"][rfi_ind]

    # Get list of bad inputs from  gains
    assert np.nanmax(np.abs(gain_reordered[rfi_ind].flatten()))>0, "all inputs have a gain of 0.0; gains appear to have zeroed out for this channel!"
    good_inputs_index = np.where(abs(gain_reordered[rfi_ind])>0.)[0]
    good_inputs_index_x = np.array([good_inputs_index[ii] for ii in
                                    range(len(good_inputs_index)) if
                                    reordered_inputs[good_inputs_index[ii]].pol=="E"])
    good_inputs_index_y = np.array([good_inputs_index[ii] for ii in
                                    range(len(good_inputs_index)) if
                                    reordered_inputs[good_inputs_index[ii]].pol=="S"])
    good_inputs_index = [good_inputs_index_x, good_inputs_index_y]
    N_good_inputs = [len(good_inputs_index[pp]) for pp in range(2)]
    from simulate_signal import fringestop_delays_vectorized
    import ch_util
    if fringestop_delays is None:
        fringestop_delays=fringestop_delays_vectorized(
            time,
            feeds=reordered_inputs,
            src_ra=ra,
            src_dec=dec,
            reference_feed=-1,
            obs=ch_util.ephemeris.chime,
            static_delays=static_delays)[:,0]

    fringestop_phase=np.exp(2j*np.pi*data.freq[rfi_ind]*fringestop_delays) #(nfreq,ninputs)


    ###########################################################################
    ############################ form visibilities ############################
    ###########################################################################
    # Find where NaNs at the end of the timestream are. Same for all inputs
    vis=[]
    good_inputs_index=np.array(good_inputs_index)
    good_inputs_index_all=np.array([good_inputs_index[0].flatten(),good_inputs_index[1].flatten()]).flatten() #
    good_inputs_index_all=np.concatenate((good_inputs_index_x, good_inputs_index_y), axis=0)

    baseband_data=data.baseband[rfi_ind][good_inputs_index_all] #ninputs, ntime

    baseband_data=baseband_data[...,frame_stop:] #remove signal when constructing visibility matrix

    baseband_data=baseband_data[:,~np.isnan(baseband_data)[-1]] #remove nans
    NN=len(baseband_data)
    vis=(mat2utvec(np.tensordot(baseband_data,
                                baseband_data.conj(),
                                axes=([-1], [-1]))) / NN)

    # Construct filter
    # The filter matrix has the form np.dot(a, b.T)

    try:
        if clean:
            print("STARTING CLEANING")
            fringestop_phasepp=np.array([fringestop_phase[i] for i in good_inputs_index_all])
            fs_phase=(np.conj(fringestop_phasepp[np.newaxis, :])*(
                        fringestop_phasepp[:, np.newaxis]))

            fs_phase=mat2utvec(fs_phase)

            S = utvec2mat(N_good_inputs[0]+N_good_inputs[1], fs_phase.conj())

            rows=S.shape[0]
            cols=S.shape[1]
            N_x=N_good_inputs[0]

            for row in range(rows):
                for col in range(cols):
                    if row<N_x:
                        if col>=N_x:
                            S[row,col]*=Ex_Ey
                        else:
                            S[row,col]*=Ex_Ex

                    else: #if row>=N_x 
                        if col<N_x:
                            S[row,col]*=Ey_Ex
                        else:
                            S[row,col]*=Ey_Ey

            F = utvec2mat(N_good_inputs[0]+N_good_inputs[1], vis) #should signal be extracted first?
            evalues, R = la.eigh(S, F) #largest eval is last element
            R_dagger = R.T.conj()
            R_dagger_inv = la.inv(R_dagger)
            a=R_dagger_inv[:, -1] #last 
            b=R_dagger[-1] #last row

            data["baseband"][rfi_ind, good_inputs_index_all] = a[:, np.newaxis] * np.sum(
                b[:, np.newaxis] * data["baseband"][rfi_ind, good_inputs_index_all],
                axis=0)[np.newaxis, :] 
    
        for pp in range(2):
            # now we fringestop after cleaning
            fringestop_phasepp=np.array([fringestop_phase[i] for i in good_inputs_index[pp]])
            fringedstop_baseband = fringestop_phasepp[:,np.newaxis]*data["baseband"][rfi_ind, good_inputs_index[pp]]
            data['tiedbeam_baseband'][rfi_ind, pp, :]= np.sum(fringedstop_baseband,axis=0)
            print(f"successfully beamformed index {rfi_ind}")


    except np.linalg.LinAlgError as err:
        time=datetime.datetime.now()
        print(
        "{0}: The following LinAlgError ocurred while constructing the RFI filter: {1}".format(
            time.strftime("%Y%m%dT%H%M%S"), err))
       
        print("will nullify data (simulated)") #do not form beamformed data for now


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
