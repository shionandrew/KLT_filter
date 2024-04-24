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

def clean_persistent_rfi(
    data,
    ra:np.ndarray, 
    dec:np.ndarray, 
    source_names:np.ndarray,
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
    obs=ephemeris.chime,
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
        try:
            clean_single_channel(
            data=data,
            ra=ra,
            dec=dec,
            obs=obs,
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
            print(f"completely finished with channel {data.freq[rfi_ind]}")

        except Exception as e:
            print(f"WARNING: COULD NOT PROCESS CHANNEL {data.freq[rfi_ind]} due to error: {e}")

    print("%s: RFI cleaning finished" %(time.strftime("%Y%m%dT%H%M%S")))
    
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
        frame_start=0
    if frame_stop is None:
        frame_stop=0
    


    # Get list of bad inputs from  gains
    assert np.nanmax(np.abs(gain_reordered[rfi_ind].flatten()))>0, "all inputs have a gain of 0.0; gains appear to have zeroed out for this channel!"
    good_inputs_index = np.where(abs(gain_reordered[rfi_ind])>0.)[0]
    good_inputs_index_x = np.array([good_inputs_index[ii] for ii in
                                    range(len(good_inputs_index)) if
                                    reordered_inputs[good_inputs_index[ii]].pol=="S"])
    good_inputs_index_y = np.array([good_inputs_index[ii] for ii in
                                    range(len(good_inputs_index)) if
                                    reordered_inputs[good_inputs_index[ii]].pol=="E"])
    good_inputs_index = [good_inputs_index_x, good_inputs_index_y]
    N_good_inputs = [len(good_inputs_index[pp]) for pp in range(2)]
    if fringestop_delays is None:
        time = (
        data["time0"]["ctime"][rfi_ind] + data["time0"]["ctime_offset"][rfi_ind]
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
            data.freq[rfi_ind],
            reordered_inputs,
            ra_from_src,
            dec_from_src,
            prod_map=prod_map,
            obs=obs,
            static_delays=static_delays,
        ).T.astype(np.complex64)[0]
    else:
        fringestop_phase=np.exp(2j*np.pi*data.freq[rfi_ind]*fringestop_delays) #(ninputs)


    ###########################################################################
    ############################ form visibilities ############################
    ###########################################################################
    # Find where NaNs at the end of the timestream are. Same for all inputs


    if clean:
        print("STARTING KLT FILTER CLEANING")
        for pp in range(2):
            # Construct filter
            # The filter matrix has the form np.dot(a, b.T)
            
            good_inputs_index_pp=good_inputs_index[pp]
            baseband_data=copy.deepcopy(data.baseband[rfi_ind][good_inputs_index_pp]) #ninputs, ntime
            print(baseband_data.shape)
            baseband_data=np.concatenate((baseband_data[...,:frame_start],baseband_data[...,frame_stop:]),axis=-1) #remove signal when constructing visibility matrix
            print(baseband_data.shape)
            baseband_data=baseband_data[:,~np.isnan(baseband_data)[-1]] #remove nans
            NN=len(baseband_data)
            vis=(mat2utvec(np.tensordot(baseband_data,
                                        baseband_data.conj(),
                                        axes=([-1], [-1]))) / NN)
            
            s_i=np.array([fringestop_phase[i] for i in good_inputs_index[pp]])
            fs_phase=mat2utvec(np.conj(s_i[np.newaxis, :])*(
                        s_i[:, np.newaxis]))
            S = utvec2mat(N_good_inputs[pp], fs_phase.conj())
            F = utvec2mat(N_good_inputs[pp], vis) #should signal be extracted first?
            c, low = cho_factor(F)
            v = np.conj(cho_solve((c, low), S))[:,-1]
            # fix phase, <v s^*>=real
            z=np.sum(v*np.conj(s_i),axis=0) 
            v/=z
            # fix amplitude, <(v s^*)^2>=<(s s^*)**2>
            v_dot_s=np.sum(np.abs(v*np.conj(s_i))**2,axis=0)
            s_dot_s=np.sum(np.abs(s_i*np.conj(s_i))**2,axis=0)
            norm_factor=s_dot_s/v_dot_s
            v*=np.sqrt(norm_factor)

            data['tiedbeam_baseband'][rfi_ind, pp, :]=np.sum(v[:,np.newaxis] * data["baseband"][rfi_ind, good_inputs_index[pp]],axis=0)

            '''fringestop_phasepp=np.array([fringestop_phase[i] for i in good_inputs_index[pp]])
            fs_phase=mat2utvec(np.conj(fringestop_phasepp[np.newaxis, :])*(
                        fringestop_phasepp[:, np.newaxis]))
            S = utvec2mat(N_good_inputs[pp], fs_phase.conj())
            F = utvec2mat(N_good_inputs[pp], vis) #should signal be extracted first?
            evalues, R = la.eigh(S, F) #largest eval is last element
            R_dagger = R.T.conj()
            R_dagger_inv = la.inv(R_dagger)
            a=R_dagger_inv[:, -1] #last 
            b=R_dagger[-1] #last row
            data["baseband"][rfi_ind, good_inputs_index[pp]] = a[:, np.newaxis] * np.sum(
                b[:, np.newaxis] * data["baseband"][rfi_ind, good_inputs_index[pp]],
                axis=0)[np.newaxis, :] #b=R
            print(f"successfully cleaned pol hand {pp} for channel index {rfi_ind}")'''

    else:
        for pp in range(2):
            # now we fringestop after cleaning
            fringestop_phase_pp=np.array([fringestop_phase[i] for i in good_inputs_index[pp]]) #ninputs 
            baseband_data_pp=copy.deepcopy(data["baseband"][rfi_ind, good_inputs_index[pp]]) #ninputs, ntime
            #fringedstop_baseband = fringestop_phase_pp[:,np.newaxis]*baseband_data_pp
            #data['tiedbeam_baseband'][rfi_ind, pp, :]= np.sum(fringedstop_baseband,axis=0)
            data['tiedbeam_baseband'][rfi_ind, pp, :]= fringestop_phase_pp @ baseband_data_pp
    print(f"successfully beamformed index {rfi_ind}")



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
