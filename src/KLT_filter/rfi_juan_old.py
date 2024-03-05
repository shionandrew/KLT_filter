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
from baseband_analysis.analysis import beamform
import baseband_analysis.core.calibration as cal
from beam_model.utils import get_equatorial_from_position, get_position_from_equatorial


def clean_persistent_rfi(bbdata, ra_target, dec_target, cyl_rotation, ps_gain_file, 
                         rfi_channels, calibrate_data=False, verbose=False,cor_inputs_db=None):
    """
    Clean baseband data from persistent RFI
    
    Parameters:
    -----------
    bbdata: core.BBdata
        baseband data
    ra_target: float
        Target (transient) RA
    dec_target: float
        Target (transient) DEC
    cyl_rotation : float
        Cylinder rotation
    ps_gain_file: str
        Point source gain file name (full path)
    rfi_bins: array
        List of persistent RFI frequencies. Only frequencies in bbdata are cleaned
    calibrate_data: bool
        If True, apply point source gains to baseband data
        before processing (data has not been calibrated).
        If False (default), do not apply point source
        gains to baseband data (data already calibrated)
    verbose: bool
        Verbose
    """

    # baseband data info
    timestamp0 = bbdata["time0"]["ctime"] + bbdata["time0"]["ctime_offset"]
    date0 = [datetime.datetime.utcfromtimestamp(t) for t in timestamp0]
    timestamp = timestamp0[:, np.newaxis] + bbdata.time[np.newaxis, :]
    N_times = len(timestamp[0])
    f_MHz = bbdata.freq
    freq_id_list = bbdata.index_map["freq"]["id"]
    N_freqs = len(f_MHz)

    # Frequency indices to clean
    rfi_indices = [i for i in range(N_freqs) if freq_id_list[i] in rfi_channels]
    rfi_channels = freq_id_list[rfi_indices]
    N_rfi_channels = len(rfi_channels)
    if verbose:
        print("{0}: Channels to clean from persistent RFI: {1} (indices {2})".format(
            time.strftime("%Y%m%dT%H%M%S"), rfi_channels, rfi_indices))

    # Get ps gains to find good inputs
    gain = cal.read_gains(ps_gain_file) # Read gains
    gain = gain[freq_id_list] # Gains for freqs in bbdata
    gain_reordered = gain[:, bbdata.input["chan_id"]] # reordered gains

    # Calibrate data
    if calibrate_data:
        if verbose:
            print("%s: Calibrating data" %(time.strftime("%Y%m%dT%H%M%S")))
        cal.apply_calibration(bbdata, cal.read_gains(ps_gain_file))

    # Read database. cor_inputs_db ordered by input ID
    if verbose:
        print("%s: Reading inputs from database" %(time.strftime("%Y%m%dT%H%M%S")))
    if cor_inputs_db is None:
        cor_inputs_db = tools.get_correlator_inputs(date0[0],
                        correlator="chime")

    # Form visibilities. Have to process each frequency separately because the good
    # inputs depend on frequency
    for i, (fb, fid) in enumerate(zip(rfi_indices, rfi_channels)):
        if verbose:
            print("\n%s: Processing RFI channel %i (%i/%i)" %(time.strftime("%Y%m%dT%H%M%S"),
                                                                    fid, i, N_rfi_channels))

        # Get list of bad inputs from ps gains
        good_inputs_index = np.where(abs(gain_reordered[fb])>0.)[0]
        good_inputs = bbdata.input["chan_id"][good_inputs_index]
        good_inputs_index_x = np.array([good_inputs_index[ii] for ii in
                                        range(len(good_inputs_index)) if
                                        cor_inputs_db[good_inputs[ii]].pol=="E"])
        good_inputs_index_y = np.array([good_inputs_index[ii] for ii in
                                        range(len(good_inputs_index)) if
                                        cor_inputs_db[good_inputs[ii]].pol=="S"])
        good_inputs_index = [good_inputs_index_x, good_inputs_index_y]
        good_inputs = [bbdata.input["chan_id"][gii] for gii in good_inputs_index]
        N_good_inputs = [len(good_inputs[pp]) for pp in range(2)]
        if verbose:
            print("%s: %i good XX inputs. %i good YY inputs" %(time.strftime("%Y%m%dT%H%M%S"),
                                                               N_good_inputs[0], N_good_inputs[1]))

        # Form visibilities
        if verbose:
            print("%s: Forming visibilities" %(time.strftime("%Y%m%dT%H%M%S")))
        N_vis = [(N_good_inputs[pp]*(N_good_inputs[pp]+1))//2 for pp in range(2)]
        # Find where NaNs at the end of the timestream are. Same for all inputs
        nan_index = np.where(np.isnan(abs(bbdata.baseband[fb, 0])))[0]
        NN = nan_index[0] if len(nan_index)>0 else N_times
        vis = [mat2utvec(np.tensordot(bbdata.baseband[fb, good_inputs_index[pp], :NN],
                                      bbdata.baseband[fb, good_inputs_index[pp], :NN].conj(),
                                      axes=([-1], [-1]))) / NN
               for pp in range(2)]
        timestamp_vis = np.array([timestamp[fb, NN//2]])
        ra_vis = ephemeris.lsa(timestamp_vis)
        ha_vis = np.radians(ra_vis - ra_target)
        prod_map = [np.empty(N_vis[pp], dtype=[("input_a", "u2"), ("input_b", "u2")])
                    for pp in range(2)]
        for pp in range(2):
            # row and col indices for upper triangular matrix
            row_index, col_index = np.triu_indices(N_good_inputs[pp])
            prod_map[pp]["input_a"], prod_map[pp]["input_b"] = row_index, col_index

        # Form position vectors
        tools.change_chime_location(rotation=cyl_rotation)
        
        feed_pos = [tools.get_feed_positions([cor_inputs_db[i] for i in good_inputs[pp]],
                                             get_zpos=False)
                    for pp in range(2)]
        baseline_x = [mat2utvec(feed_pos[pp][:, 0][np.newaxis, :] -
                                feed_pos[pp][:, 0][:, np.newaxis])
                      for pp in range(2)]
        baseline_y = [mat2utvec(feed_pos[pp][:, 1][np.newaxis, :] -
                                feed_pos[pp][:, 1][:, np.newaxis])
                      for pp in range(2)]
        wv = scipy.constants.c * 1e-6 / f_MHz[fb]
        u = [baseline_x[pp] / wv for pp in range(2)]
        v = [baseline_y[pp] / wv for pp in range(2)]

        # Construct filter
        if verbose:
            print("%s: Constructing RFI filter" %(time.strftime("%Y%m%dT%H%M%S")))
        # The filter matrix has the form np.dot(a, b.T)
        a = []
        b = []
        try:
            for pp in range(2):
                fs_phase = tools.fringestop_phase(ha_vis, np.radians(ephemeris.CHIMELATITUDE),
                                                  np.radians(dec_target), u[pp], v[pp], 0.)
                print("fs_phase")
                print(fs_phase[:10])
                print(fs_phase[-10:])
                S = utvec2mat(N_good_inputs[pp], fs_phase.conj())
                F = utvec2mat(N_good_inputs[pp], vis[pp])
                evalues, P = la.eigh(S, F)
                P_dagger = P.T.conj()
                P_dagger_inv = la.inv(P_dagger)
                a.append(P_dagger_inv[:, -1])
                b.append(P_dagger[-1])
        except np.linalg.LinAlgError as err:
            print(
            "{0}: The following LinAlgError ocurred while constructing the RFI filter: {1}".format(
                time.strftime("%Y%m%dT%H%M%S"), err))
            continue

        # Filter baseband
        if verbose:
            print("%s: Filtering baseband data" %(time.strftime("%Y%m%dT%H%M%S")))
        for pp in range(2):
            # d1 = np.dot(np.dot(a, b.T), d)
            bbdata["baseband"][fb, good_inputs_index[pp]] = a[pp][:, np.newaxis] * np.sum(
                b[pp][:, np.newaxis] * bbdata["baseband"][fb, good_inputs_index[pp]],
                axis=0)[np.newaxis, :]

    if verbose:
        print("%s: RFI cleaning finished" %(time.strftime("%Y%m%dT%H%M%S")))

    return


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


def tied_array(data, ra, dec, correlator_inputs=None, TOA_400=None,
               telescope_rotation=None, DM=None, source_name=None,
               ps_gain_file=None, rfi_channels=None,
               verbose=False):
    """
    Same as beamform.tied_array but with the option to attempt persistent RFI cleaning
    on selected frequency bins. Only one beam is allowed (ra, dec are float)
    
    Parameters:
    -----------
    data: core.BBdata
        (Calibrated) baseband data. Same as in beamform.tied_array
    ra: float
        Target RA.
    dec: float
        Target DEC.
    correlator_inputs: list
        List of correlator inputs. Same as in beamform.tied_array
    TOA_400: float
        Time of arrival. Same as in beamform.tied_array
    telescope_rotation : float
        Cylinder rotation. Same as in beamform.tied_array
    DM: float
        Dispersion Measure. Same as in beamform.tied_array
    source_name: str
        Source name. Same as in beamform.tied_array
    ps_gain_file: str
        Point source gain file name (full path).
        Same as in clean_persistent_rfi
    rfi_bins: array
        List of persistent RFI frequencies. Only frequencies in bbdata are cleaned
        Same as in clean_persistent_rfi
    verbose: bool
        Verbose.
        Same as in clean_persistent_rfi
    """
    
    if rfi_channels is not None:
        clean_persistent_rfi(data, ra, dec, telescope_rotation, ps_gain_file,
            rfi_channels, verbose=verbose)
    beamform.tied_array(data, [ra], [dec], correlator_inputs, TOA_400,
                        telescope_rotation, DM, source_name)
    return


def bbevent2tiedarray(bbdata_path, ra, dec, correlator_inputs=None, TOA_400=None,
               telescope_rotation=None, DM=None, source_name=None,
               ps_gain_file=None, rfi_channels=None,
               verbose=True, files_per_read=8, freqs_to_process=None, out_file=None):
    """
    Form tied array beam from a baseband dataset.

    Parameters:
    -----------

    bbdata_path: str
        Path to baseband dataset
    files_per_read: int
        Number of baseband files to read simultaneously (for memory saving purposes)
    freqs_to_process: list
        List of frequency chahnels to read

    All other parameters as in tied_array
    """

    datafiles = glob.glob(os.path.join(bbdata_path, "*.h5"))
    freqs = np.array([int(re.split("\.|_", os.path.split(f)[1])[2]) for f in datafiles])
    if freqs_to_process is None:
        freqs_to_process = freqs.copy()
    else:
        freqs_to_process = np.asarray(freqs_to_process)
    # Check freqs to process are in the data
    mask = np.isin(freqs_to_process, freqs)
    if not np.all(mask): 
        raise OSError("{0}: Frequency channels {1} not in {2}".format(
            time.strftime("%Y%m%dT%H%M%S"), freqs_to_process[~mask], bbdata_path))
    # Select files to read
    sorter = np.argsort(freqs)
    f_ind = np.searchsorted(freqs, freqs_to_process, sorter=sorter)
    freqs_to_process = freqs[sorter[f_ind]]
    datafiles_to_process = [datafiles[i] for i in sorter[f_ind]]
    # Process selected files
    Nfiles = len(freqs_to_process)
    Nreads = int(np.ceil(Nfiles // files_per_read))
    beamformed_data_list = []
    for i in range(Nreads):
        read_files = datafiles_to_process[i*files_per_read:(i+1)*files_per_read]
        read_freqs = freqs_to_process[i*files_per_read:(i+1)*files_per_read]
        if verbose:
            print("{0}: Processing frequency channels: {1}".format(time.strftime("%Y%m%dT%H%M%S"),
                read_freqs))
        data = bbcore.BBData.from_acq_h5(read_files)
        if verbose:
            print("%s: Applying point source calibration" %(time.strftime("%Y%m%dT%H%M%S")))
        cal.apply_calibration(data, cal.read_gains(ps_gain_file))
        tied_array(data, ra, dec, correlator_inputs, TOA_400, telescope_rotation, DM, source_name,
            ps_gain_file, rfi_channels, verbose)
        del data["baseband"] # delete baseband to save memory
        beamformed_data_list.append(data)
        if verbose:
            print("{0}: {1}/{2} frequencies processed".format(time.strftime("%Y%m%dT%H%M%S"), 
                i*files_per_read+len(read_freqs), Nfiles))
    beamformed_data = bbcore.concatenate(beamformed_data_list)
    if out_file is None:
        return beamformed_data
    else:
        beamformed_data.save(out_file)
        beamformed_data.close()
        if verbose:
            print("{0}: Data saved to {1}".format(time.strftime("%Y%m%dT%H%M%S"), out_file))

            
            
adc_sampling_freq = float(800e6)

# Number of samples in the inital FFT in the F-engine.
fpga_num_samp_fft = 2048
# f-engine parameters for alias sampling in second Nyquist zone.
k_DM = 1.0 / 2.41e-4
fpga_num_freq = fpga_num_samp_fft // 2
fpga_freq0_mhz = adc_sampling_freq / 1e6
fpga_delta_freq_mhz = -adc_sampling_freq / 2 / fpga_num_freq / 1e6

def delay_across_the_band(DM, freq_low=400, freq_high=800):
    """
    Return the delay in seconds caused by dispersion, given
    a Dispersion Measure (DM) in cm-3 pc, and the emitted
    frequency (freq_emitted) of the pulsar in MHz.

    Parameters
    ----------
    DM: float
    freq_low: float
        Lowest observing frequency in MHz.
    freq_high: float
        Highest observing frequency in MHz.

    """
    return k_DM * DM * (1.0 / freq_low ** 2 - 1.0 / freq_high ** 2)

def get_nDM(DM, f0, dt=2.56e-6):
    df = abs(fpga_delta_freq_mhz)
    return (
        int(
            (
                np.ceil(
                    delay_across_the_band(DM, f0 - df / 2, f0 + df / 2)
                    / dt
                )
            ).max()
        )
        // 2
        + 1
    )
def pow2(x):
    return (1 << (x - 1).bit_length()) - x

def coherent_dedisp(data, DM=None,inversion=False):
    # data is a BBdata instance
    # Sampling frequency, in Hz.
    from scipy.fftpack import fft, ifft
    if DM is None:
        DM = data["tiedbeam_baseband"].attrs["DM"]
    f0 = data.index_map["freq"]["centre"][:, np.newaxis]
    df = abs(fpga_delta_freq_mhz)
    ctime = data["time0"]["fpga_count"].astype(float) - data["time0"]["fpga_count"][
        -1
    ].astype(float)
    ctime = data.attrs["delta_time"] * ctime[:, np.newaxis]
    nDM = get_nDM(DM, f0, dt=data.attrs["delta_time"])
    fill = pow2(data.ntime + nDM * 2)
    shape = list(data["tiedbeam_baseband"].shape)
    shape[-1] += fill + nDM * 2
    t = np.zeros(shape, dtype=np.complex64)
    ntime = data.ntime
    matrix = data["tiedbeam_baseband"][:]
    mask = np.isnan(matrix)
    matrix[mask] = 0
    t[..., fill // 2 : fill // 2 + ntime] = matrix
    f = np.fft.fftfreq(t.shape[-1], d=data.attrs["delta_time"] * 1e6)
    H = np.exp(
        + 2j * np.pi * 1e6 * k_DM * DM * f ** 2 / (f + f0) / f0 ** 2  # Intrachannel de-dispersion
#        - 2j * np.pi * 1e6 * (f + f0) * (ctime - common_utils.k_DM * DM * (1 / f0 ** 2 - 1 / f0[-1] ** 2))  # Time shift to align the signal
    )[:, np.newaxis]
    dedispersed_array = ifft(fft(t) * H)[..., fill // 2 : fill // 2 + ntime]
    dedispersed_array = ifft(fft(t) * H)[..., fill // 2 : fill // 2 + ntime]
    dedispersed_array[mask] = np.nan
    return dedispersed_array






import datetime
import logging
import os

import chimedb.core
import h5py
import numpy as np
import pandas as pd
import tenacity
from ch_util import ephemeris, tools
from scipy import constants, fftpack



# Logging Config
LOGGING_CONFIG = {}
logging_format = "[%(asctime)s] %(process)d-%(levelname)s "
logging_format += "%(module)s::%(funcName)s():l%(lineno)d: "
logging_format += "%(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO)
log = logging.getLogger()

# Correct for CHIME rotation
if tools.CHIMEAntenna._rotation == 0:
    tools.change_chime_location(rotation=-0.071)



def tied_array(
    data,
    ra,
    dec,
    correlator_inputs=None,
    TOA_400=None,
    telescope_rotation=None,
    DM=None,
    source_name=None,
):
    ra, dec = (
        np.asarray(ra, dtype=float).reshape(-1),
        np.asarray(dec, dtype=float).reshape(-1),
    )
    if ra.size != dec.size:
        raise ValueError("RA and Dec arrays must have the same size")

    if telescope_rotation is not None:
        tools.change_chime_location(rotation=telescope_rotation)
    log.info(f"    -- The telescope rotation is {tools.CHIMEAntenna._rotation} deg")
    if (TOA_400 is not None) and (DM is None):
        raise ValueError("DM must be provided together with TOA_400")

    log.info("    -- Loading values")
    # Get representation of the correlator inputs that includes thier type and
    # position.
    if correlator_inputs is None:
        inputs = inputs_from_data(data)
    else:
        inputs = list(correlator_inputs)

    # Reorder to match the data
    converted_data_inputs = data.input.astype(
        [("chan_id", "<u2"), ("correlator_input", "U32")]
    )
    reordered_inputs = tools.reorder_correlator_inputs(converted_data_inputs, inputs)
    prod_map = np.empty(len(data.input), dtype=[("input_a", "u2"), ("input_b", "u2")])
    prod_map["input_a"] = np.arange(len(data.input))
    prod_map["input_b"] = 0

    nfreq = data.nfreq
    ntime = data.ntime
    ninput = data.ninput
    npointings = ra.size
    nbeam = npointings * 2  # 2 polarizations.

    pol_mask = np.zeros((2, ninput), dtype=np.int32)
    for ii in range(ninput):
        if type(reordered_inputs[ii]) != tools.CHIMEAntenna:
            continue
        elif reordered_inputs[ii].pol == "S":
            pol_mask[0, ii] = 1
        elif reordered_inputs[ii].pol == "E":
            pol_mask[1, ii] = 1

    log.info("    -- Creating dataset")
    d = data.create_dataset(
        "tiedbeam_baseband", shape=(nfreq, nbeam, ntime), dtype=data.baseband.dtype
    )
    beam_axis = _make_beammap(data, nbeam)
    d.attrs["axis"] = ["freq", beam_axis, "time"]
    d.attrs["fill_value"] = np.complex64(np.nan)

    ib_dtype = [
        ("ra", float),
        ("dec", float),
        ("x_400MHz", float),
        ("y_400MHz", float),
        ("pol", "S1"),
    ]
    if source_name is not None:
        ib_dtype.append(("source_name", "<S20"))
    ib = np.empty(nbeam, dtype=ib_dtype)
    ib["ra"] = (ra[:, None] * np.ones(2, dtype=ra.dtype)).flat
    ib["dec"] = (dec[:, None] * np.ones(2, dtype=dec.dtype)).flat

    # Calculate local coordinates
    if TOA_400 is None:
        time = data["time0"]["ctime"][-1] + data["time0"]["ctime_offset"][-1]
        time = time + np.mean(data.index_map["time"]["offset_s"])
    else:
        time = TOA_400
    x, y = get_position_from_equatorial(ra, dec, time, telescope_rotation_angle=None)
    ib["x_400MHz"] = (x[:, None] * np.ones(2, dtype=x.dtype)).flat
    ib["y_400MHz"] = (y[:, None] * np.ones(2, dtype=y.dtype)).flat
    ib["pol"] = ["S", "E"] * npointings
    if source_name is not None:
        ib["source_name"] = [y for x in source_name for y in (x,) * 2]
    loc = data.create_dataset("tiedbeam_locations", data=ib)
    loc.attrs["axis"] = [beam_axis]

    log.info("    -- Processing each channel")
    for jjfreq in range(nfreq):
        # Time is frequency dependent due to substantial dispersion delay
        if TOA_400 is None:
            time = (
                data["time0"]["ctime"][jjfreq] + data["time0"]["ctime_offset"][jjfreq]
            )
            time = time + np.mean(data.index_map["time"]["offset_s"])
        else:
            time = TOA_400 + common_utils.delay_across_the_band(
                DM, data.freq[jjfreq], 400.0
            )

        log.info(f"      --- Frequency n. {jjfreq}: Exctracting channel")
        array_baseband = data.baseband[jjfreq : jjfreq + 1]

        log.info(f"      --- Frequency n. {jjfreq}: Forming single beams")
        for iipoint in range(npointings):
            src = ephemeris.skyfield_star_from_ra_dec(ra[iipoint], dec[iipoint])
            log.debug(f"        ---- Pointing n. {iipoint}: Start fringestopping")
            # Here we assume the beamforming phase does not depend on time, since
            # baseband dumps are typically very short (less than a second at a single frequency).
            # Thus we only need to calculate one set of phases.
            fringestop_phase = (
                tools.fringestop_time(
                    np.ones(array_baseband.shape[:-1] + (1,), array_baseband.dtype),
                    np.array([time]),
                    data.freq[[jjfreq]],
                    reordered_inputs,
                    src,
                    prod_map=prod_map,
                )
                .astype(np.complex64)
                .conj()
            )

            log.debug(f"        ---- Pointing n. {iipoint}: Start summing")
            beams = tiedbeam_apply_phase_sum(
                array_baseband[0], fringestop_phase[0, :, 0], pol_mask
            )

            log.debug(f"        ---- Pointing n. {iipoint}: Writing to database")
            d[jjfreq, 2 * iipoint : 2 * (iipoint + 1), :] = beams

    d.attrs["conjugate_beamform"] = int(1)
    log.info("    -- Beamforming finished")
    return


def ra_to_deg(ra_string):
    """
    From PRESTO psr_utils

    ra_to_rad(ar_string):
       Given a string containing RA information as
       'hh:mm:ss.ssss', return the equivalent decimal
       degrees.
    """
    h, m, s = ra_string.split(":")

    hour = int(h)
    minute = int(m)
    second = float(s)

    if hour < 0.0:
        sign = -1
    else:
        sign = 1
    return (
        sign
        * 0.004166666666666667
        * (60.0 * (60.0 * np.fabs(hour) + np.fabs(minute)) + np.fabs(second))
    )


def dec_to_deg(dec_string):
    """
    From PRESTO psr_utils

    dec_to_rad(dec_string):
       Given a string containing DEC information as
       'dd:mm:ss.ssss', return the equivalent decimal
       degrees.
    """
    d, m, s = dec_string.split(":")
    if "-" in d and int(d) == 0:
        m, s = "-" + m, "-" + s

    degree = int(d)
    minute = int(m)
    second = float(s)

    if degree < 0.0:
        sign = -1
    elif degree == 0.0 and (minute < 0.0 or degree < 0.0):
        sign = -1
    else:
        sign = 1
    return (
        sign
        * 0.0002777777777777778
        * (60.0 * (60.0 * np.fabs(degree) + np.fabs(minute)) + np.fabs(second))
    )


@tenacity.retry(
    wait=tenacity.wait_random(5, 15),
    stop=tenacity.stop_after_delay(120),
    reraise=True,
    retry=tenacity.retry_if_exception_type(chimedb.core.exceptions.ConnectionError),
    before_sleep=tenacity.before_sleep_log(log, logging.WARNING),
)
def inputs_from_data(data):
    date_unix = data["time0"]["ctime"][0] + data["time0"]["ctime_offset"][0]
    date = datetime.datetime.utcfromtimestamp(date_unix)
    # 'FCC' should ultimately be set from the data too, so this works with the
    # pathfinder.
    inputs = tools.get_correlator_inputs(date, correlator="FCC")
    return inputs


def _make_beammap(data, nbeams):
    suffix = ""
    imap_names = list(data.index_map.keys())
    for map_num in range(10):
        beam_axis = "beam" + suffix
        if beam_axis not in imap_names:
            break
        else:
            suffix = str(map_num)
    else:
        raise RuntimeError("More than 10 beam maps.")
    beammap = np.arange(nbeams, dtype=int)
    data.create_index_map(beam_axis, beammap)
    return beam_axis



def tiedbeam_apply_phase_sum(baseband,phases,masks,
        ):
    ninput = baseband.shape[0]
    ntime = baseband.shape[1]
    nmasks = masks.shape[0]
    out = np.zeros((nmasks, ntime),dtype=np.complex64)

    tmp=0.0j
    
    for iit in range(ntime):#, nogil=True):
        for jjm in range(nmasks):
            for kki in range(ninput):
                tmp = baseband[kki, iit]
                tmp = tmp * phases[kki]
                tmp = tmp * masks[jjm, kki]
                out[jjm, iit] = out[jjm, iit] + tmp
    return out
