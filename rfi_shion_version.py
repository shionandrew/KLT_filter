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
from typing import List
import baseband_analysis.core.calibration as cal
from baseband_analysis.analysis.beamform import fringestop_time_vectorized

def clean_persistent_rfi(
    data,
    ra:np.ndarray, 
    dec:np.ndarray, 
    ps_gain_file:str, 
    inputs:List,
    static_delays:bool,
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
    timestamp0 = data["time0"]["ctime"] + data["time0"]["ctime_offset"]
    date0 = [datetime.datetime.utcfromtimestamp(t) for t in timestamp0]
    timestamp = timestamp0[:, np.newaxis] + data.time[np.newaxis, :]
    N_times = len(timestamp[0])
    f_MHz = data.freq
    N_freqs = len(f_MHz)

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

    print("STARTING CLEANING")
    for rfi_ind in range(N_freqs):
        try:
            clean_single_channel(
            data=data,
            ra=ra,
            dec=dec,
            obs=obs,gain_reordered=gain_reordered,
            rfi_ind=rfi_ind,
            prod_map=prod_map,
            reordered_inputs=reordered_inputs,
            static_delays=static_delays
        )
            print(f"successfully cleaned channel {data.freq[rfi_ind]}")

        except Exception as e:
            print(f"WARNING: COULD NOT PROCESS CHANNEL {data.freq[rfi_ind]} due to error: {e}")

    print("%s: RFI cleaning finished" %(time.strftime("%Y%m%dT%H%M%S")))

    return

def clean_single_channel(
    data,
    ra,
    dec,
    obs,
    rfi_ind:int,
    gain_reordered,
    prod_map,
    static_delays,
    reordered_inputs:List
):

    time = (
            data["time0"]["ctime"][rfi_ind] + data["time0"]["ctime_offset"][rfi_ind]
        )
    time = time + data.index_map["time"]["offset_s"][rfi_ind]
    ra_from_src = np.empty(ra.shape[0])
    dec_from_src = np.empty(dec.shape[0])
    for iipoint in range(len(ra)):
        src = ephemeris.skyfield_star_from_ra_dec(ra[iipoint], dec[iipoint])
        (
            ra_from_src[iipoint],
            dec_from_src[iipoint],
        ) = ephemeris.object_coords(src, time, obs=obs)

    # Get list of bad inputs from  gains
    assert np.nanmax(np.abs(gain_reordered[rfi_ind].flatten()))>0, "gains appear to have zeroed out rfi channels!"
    good_inputs_index = np.where(abs(gain_reordered[rfi_ind])>0.)[0]
    good_inputs_index_x = np.array([good_inputs_index[ii] for ii in
                                    range(len(good_inputs_index)) if
                                    reordered_inputs[good_inputs_index[ii]].pol=="E"])
    good_inputs_index_y = np.array([good_inputs_index[ii] for ii in
                                    range(len(good_inputs_index)) if
                                    reordered_inputs[good_inputs_index[ii]].pol=="S"])
    good_inputs_index = [good_inputs_index_x, good_inputs_index_y]
    N_good_inputs = [len(good_inputs_index[pp]) for pp in range(2)]
        
    fringestop_phase = fringestop_time_vectorized(
        time,
        data.freq[rfi_ind],
        reordered_inputs,
        ra_from_src,
        dec_from_src,
        prod_map=prod_map,
        obs=obs,
        static_delays=static_delays,
    ).T.astype(np.complex64)[0]

    ###########################################################################
    ############################ form visibilities ############################
    ###########################################################################
    # Find where NaNs at the end of the timestream are. Same for all inputs
    vis=[]
    for pp in range(2):
        good_inputs_index_pp=good_inputs_index[pp]
        baseband_data=data.baseband[rfi_ind][good_inputs_index_pp] #ninputs, ntime
        baseband_data=baseband_data[:,~np.isnan(baseband_data)[-1]] #remove nans
        NN=len(baseband_data)
        vis.append(mat2utvec(np.tensordot(baseband_data,
                                    baseband_data.conj(),
                                    axes=([-1], [-1]))) / NN)

    # Construct filter
    # The filter matrix has the form np.dot(a, b.T)
    a = []
    b = []

    try:
        for pp in range(2):
            fringestop_phasepp=np.array([fringestop_phase[i] for i in good_inputs_index[pp]])
            fs_phase=mat2utvec(np.conj(fringestop_phasepp[np.newaxis, :])*(
                        fringestop_phasepp[:, np.newaxis]))
            S = utvec2mat(N_good_inputs[pp], fs_phase.conj())
            F = utvec2mat(N_good_inputs[pp], vis[pp]) #should signal be extracted first?
            evalues, R = la.eigh(S, F) #largest eval is last element
            R_dagger = R.T.conj()
            R_dagger_inv = la.inv(R_dagger)
            a.append(R_dagger_inv[:, -1]) #last 
            b.append(R_dagger[-1]) #last row

        for pp in range(2):
            data["baseband"][rfi_ind, good_inputs_index[pp]] = a[pp][:, np.newaxis] * np.sum(
                b[pp][:, np.newaxis] * data["baseband"][rfi_ind, good_inputs_index[pp]],
                axis=0)[np.newaxis, :] #b=R

    except np.linalg.LinAlgError as err:
        time=datetime.datetime.now()
        print(
        "{0}: The following LinAlgError ocurred while constructing the RFI filter: {1}".format(
            time.strftime("%Y%m%dT%H%M%S"), err))

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
