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
from coda.core import VLBIVis
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

from outriggers_vlbi_pipeline.calibration import get_calibrator_visibilities, make_calibrated_visibilities
from KLT_filter.clean_correlation import clean_and_form_singlebeams

database=kko_events_database
version=current_version

def target_singlebeam_exists(event_id, telescope, version):
    try: 
        target_singlebeam = glob.glob(f'/arc/projects/chime_frb/vlbi/rev{version}/*/*/*/{event_id}/target_singlebeams/*{telescope}.h5')[0]
        return True
    except Exception as e: 
        return False

def run_baseband_localization_pipeline(eid):
    from skaha.session import Session
    import nest_asyncio
    import os
    import time
    nest_asyncio.apply()
    s = Session()
    sids = []
    payload = {
            'name': 'baseband-kwmdev',
            'image': 'images.canfar.net/chimefrb/baseband-analysis:kwmdev',
            'cores': 4,
            'ram': 16,
            'kind': 'headless',
            'cmd': '/arc/projects/chime_frb/kiyo/process_event.sh',
            'args': " ".join([str(eid), "--no-pull"]),
            'env': {'SITE': 'canfar', 'CHIME_FRB_ACCESS_TOKEN': os.environ["CHIME_FRB_ACCESS_TOKEN"], 'CHIME_FRB_REFRESH_TOKEN': os.environ["CHIME_FRB_REFRESH_TOKEN"]},
            'replicas': 1,
    }
    sid = s.create(**payload)
    sids = sids + [(eid, sid[0])]
    time.sleep(100)

        
if __name__=='__main__': 
    
    parser = argparse.ArgumentParser("rfi Executable")
    parser.add_argument("--clean", help="clean", type=str,default='False')
    cmdargs = parser.parse_args()
    clean=cmdargs.clean
    if clean=='False' or clean=='false':
        clean=False
    else:
        clean=True
    
    if clean:
        print("WILL CLEAN")
        tag='rfi_cleaned'
    else:
        print("WILL NOT CLEAN")
        tag='unclean'
    database=kko_events_database

    events=[350256958]#[350258229,350258482,350258834,350259263,350259503]#[350256958,350257224,350257489,350257811]#350258229,350258482,350258834,350259263,350259503]#
    for event_id in events:
        raw_data_dir=f'/arc/projects/chime_frb/data/gbo/astro_{event_id}/*.h5'
        ctime=1704594691.5901184
        cal_h5 = '/arc/home/shiona/gains/gain_20240104T011813.833671Z_casa.h5'

        pulsar_name='B0136+57'
        ratrue,dectrue=get_true_pulsar_pos(pulsar_name,ctime=ctime)
        
        ra_cal=24.2441450135833
        dec_cal=47.8580833533333
        cal_name='J013658.5+475129'

        ra_cal2=24.832133450058727
        dec_cal2=33.15974373333333
        cal_name2='J0137+3309'

        ras=np.array([ratrue,ra_cal,ra_cal2])
        decs=np.array([dectrue,dec_cal,dec_cal2])
        src_names=np.array([pulsar_name,cal_name,cal_name2])


        log_folder='/arc/home/shiona/'
        log_file_event = os.path.join(log_folder,'test.log')

        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
            handlers=[logging.FileHandler(log_file_event),  # Output log messages to a log file specific to this event
            logging.StreamHandler()],  # Output log messages to the console
            force=True
        )
        logging.info(f"writing to {log_file_event}")
        telescope='gbo'

        out_files=clean_and_form_singlebeams(
            event_id=event_id,
            ras=ras, #array of floats
            decs=decs, #array of floats
            src_names=src_names, #array of strings
            telescope=telescope,
            raw_data_dir=raw_data_dir,
            cal_h5=cal_h5,
            ctime=ctime,
            version=version,
            events_database=database,
            source_types=['target','calibrator','calibrator'],
            tag=tag,
            clean=clean,
        )
        print(out_files)