import sys
import numpy as np
from glob import glob
import copy
from typing import Optional
import logging
import baseband_analysis.core.calibration as cal
import os
from outriggers_vlbi_pipeline.known_calibrators import get_true_pulsar_pos
from ch_util import tools
from KLT_filter.rfi_shion_version import beamform_clean_single_channel
import baseband_analysis.core.bbdata as bbdata
from baseband_analysis import utilities
from baseband_analysis.analysis import beamform
import baseband_analysis.core.calibration as cal
from datetime import datetime
from outriggers_vlbi_pipeline.vlbi_pipeline_config import (
    VeryBasicBackend,
    kko_backend,
    gbo_backend,
    chime_backend,
    gbo_backend,
    valid_telescopes,
    current_version,
    kko_events_database,
    get_file_path
)
from baseband_analysis.core.bbdata import BBData
from glob import glob
from caput import memh5
from beam_model.utils import get_equatorial_from_position, get_position_from_equatorial

from outriggers_vlbi_pipeline.query_database import get_event_data, update_event_status,get_full_filepath,find_files
from outriggers_vlbi_pipeline.multibeamform import get_calfiles
from ch_util import ephemeris, tools
from typing import *

def clean_and_form_singlebeams(
    event_id,
    ras, #array of floats
    decs, #array of floats
    src_names, #array of strings
    telescope:str,
    raw_data_dir:str,
    cal_h5:str,
    ctime:float,
    source_types: Optional[np.ndarray] = None,
    version: str = current_version,
    events_database=kko_events_database,
    tag:str='',
    clean:bool=True,
    **pol_kwargs,
    ):
    if source_types is None:
        logging.info("WILL ASSUME FIRST INDEX IS TARGET")
        source_types=['target']
        for source in ras[1:]:
            source_types.append('calibrator')
    if telescope=='kko':
        backend=kko_backend
        reference_feed = tools.ArrayAntenna(
                id=-1, slot=-1, powered=True, flag=True, pos=[0, 0, 0], delay=0
        ) 
        backend.static_delays=False
        static_delays=False
    if telescope=='chime':
        backend=chime_backend
        reference_feed=tools.CHIMEAntenna(
                id=-1, slot=-1, powered=True, flag=True, pos=[0, 0, 0]
        )
        backend.static_delays=True

    datafiles_all=glob(raw_data_dir)
    out_files=[]
    for i in range(len(ras)):
        source_type=source_types[i]
        source_name=src_names[i]
        ra=ras[i]
        dec=decs[i]
        output_file_dir=get_full_filepath(
            event_id=event_id,data_type='singlebeams',
            source_type=source_type,telescope=telescope,
            version=version,events_database=events_database)
        os.umask(0)
        os.makedirs(
            output_file_dir, exist_ok=True, mode=0o777
        )  # if output directory doesn't exist, create it

        out_file=f'{output_file_dir}{event_id}_{telescope}_singlebeam_{source_name}{tag}.h5'
        logging.info(f"Will save singlebeam to {out_file}")

        datas=[]
        date = datetime.utcfromtimestamp(ctime)
        inputs = tools.get_correlator_inputs(date, correlator=backend.correlator)
        chunk_size=8

        for i in range(len(datafiles_all)//chunk_size):
            logging.info(f"{i*chunk_size}-{i*chunk_size+chunk_size} of {len(datafiles_all)} for telescope {telescope}!")
            data = bbdata.BBData.from_acq_h5(datafiles_all[i*chunk_size:i*chunk_size+chunk_size])

            gains = cal.read_gains(cal_h5)
            cal.apply_calibration(data, gains, inputs=inputs)

            beamform_clean_rfi(
                data=data, 
                ra=np.array([ra]),
                    dec=np.array([dec]), 
                source_names=np.array([source_name]),
                    gains=gains,
                    inputs=inputs,
                reference_feed=reference_feed,
                static_delays=backend.static_delays,
                obs=backend.obs,
                clean=clean,
                **pol_kwargs)

            del data["baseband"]
            if 'input_signal' in list(data.keys()):
                del data['input_signal']
            datas.append(data)
        beamformed_rfi_cleaned = bbdata.concatenate(datas)

        beamformed_rfi_cleaned.save(out_file)
        logging.info(f"saving to {out_file}")
        beamformed_rfi_cleaned.close()
        out_files.append(out_file)
    return out_files

def reorder_gains(data,gains):
    # Get gains (nfreq,ninputs) (ordered to bbdata)
    freq_id_list = data.index_map["freq"]["id"]
    gain = gains[freq_id_list] #only get relevent frequencies
    return gain[:, data.input["chan_id"]] # reordered gains by channel input

def beamform_clean_rfi(
    data,
    ra:np.ndarray, 
    dec:np.ndarray, 
    source_names:np.ndarray,
    gains:np.ndarray, 
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
    **pol_kwargs,
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


    gain_reordered=reorder_gains(data=data,gains=gains)
    # order feed positions (ordered to bbdata)
    converted_data_inputs = data.input.astype(
            [("chan_id", "<u2"), ("correlator_input", "U32")]
        )
    reordered_inputs = tools.reorder_correlator_inputs(
                converted_data_inputs, inputs
            )

    reordered_inputs.append(reference_feed)

    nfreq=data.baseband.shape[0]
    npointing=1
    npol=2
    ntime=data.baseband.shape[-1]

    s_f=data.create_dataset(
        "S_F", shape=(nfreq, npointing*npol), dtype=data.baseband.dtype
    )
    memh5.copyattrs(data["baseband"].attrs, s_f.attrs)
    s_f.attrs["axis"] = ["freq", "beam"]

    tiedbeam_baseband = data.create_dataset(
        "tiedbeam_baseband", shape=(nfreq, npointing*npol, ntime), dtype=data.baseband.dtype
    )
    
    memh5.copyattrs(data["baseband"].attrs, tiedbeam_baseband.attrs)
    tiedbeam_baseband.attrs["axis"] = ["freq", "beam", "time"]
    tiedbeam_baseband.attrs["conjugate_beamform"] = int(1)


    for freq_index in range(N_freqs):
        if True:#try:
            beamform_clean_single_channel(
            data=data,
            ra=ra,
            dec=dec,
            obs=obs,
            freq_index=freq_index,
            gains=gain_reordered,
            fringestop_delays=fringestop_delays,
            reordered_inputs=reordered_inputs,
            static_delays=static_delays,
            clean=clean,
            frame_start=frame_start,
            frame_stop=frame_stop,
            **pol_kwargs,
        )
            print(f"successfully cleaned channel {data.freq[freq_index]}")

        else:#except Exception as e:
            ct=datetime.now()
            print(f"{ct.strftime('%Y%m%dT%H%M%S')} WARNING: COULD NOT PROCESS CHANNEL {data.freq[freq_index]} due to error: {e}")
    ct=datetime.now()
    print("%s: RFI cleaning finished" %(ct.strftime("%Y%m%dT%H%M%S")))
    ###REINSTATE!
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