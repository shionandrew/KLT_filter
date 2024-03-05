import sys
import numpy as np
from glob import glob

from rfi_shion_version import clean_persistent_rfi
from typing import Optional
import logging
import baseband_analysis.core.calibration as cal
import os
from outriggers_vlbi_pipeline.known_calibrators import get_true_pulsar_pos
from ch_util import tools

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

from outriggers_vlbi_pipeline.query_database import get_event_data, update_event_status,get_full_filepath,find_files
from outriggers_vlbi_pipeline.multibeamform import get_calfiles


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

            if clean:
                clean_persistent_rfi(
                    data=data, ra=np.array([ra]), dec=np.array([dec]), ps_gain_file=cal_h5,inputs=inputs,
                    reference_feed=reference_feed,static_delays=backend.static_delays,
                    obs=backend.obs)

            beamform.tied_array(
                    data,
                    ra=np.array([ra]),
                    dec=np.array([dec]),
                    source_name=np.array([source_name]),
                    correlator_inputs=inputs,
                    obs=backend.obs,
                    reference_feed=reference_feed,
                    static_delays=backend.static_delays,#telescope_rotation=telescope_rot
                )

            del data["baseband"]
            datas.append(data)
        beamformed_rfi_cleaned = bbdata.concatenate(datas)

        beamformed_rfi_cleaned.save(out_file)
        logging.info(f"saving to {out_file}")
        beamformed_rfi_cleaned.close()