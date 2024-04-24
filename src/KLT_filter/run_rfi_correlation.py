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
from KLT_filter.interpolate_gains import make_interpolated_gain_file

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
    parser = argparse.ArgumentParser("Localisation Pipeline Executable")
    parser.add_argument("--event_id", help="event id", type=int,default=0)
    parser.add_argument("--nstart", help="nstart", type=int,default=0)
    parser.add_argument("--nstop", help="nstop", type=int,default=0)
    parser.add_argument("--clean", help="apply rfi filter, True or False", type=str)
    parser.add_argument("--off_pointing", help="point to true position or bb position, True or False", type=str)
    parser.add_argument("--interpolate", help="use interpolated gains, True or False", type=str,default='False')
    parser.add_argument("--del_data", help="del_data", type=str,default='True')
    parser.add_argument("--test",type=int,default=0)

    cmdargs = parser.parse_args()
    clean=cmdargs.clean
    del_data=cmdargs.del_data
    interpolate=cmdargs.interpolate
    event_id=cmdargs.event_id
    nstart=cmdargs.nstart
    nstop=cmdargs.nstop
    off_pointing=cmdargs.off_pointing
    test=cmdargs.test

    if del_data=="True":
        del_data=True
    else:
        del_data=False
        print("WILL NOT DELETE DATA")
    if clean=='True':
        clean=True
    elif clean=='False':
        clean=False
    else:
        raise AssertionError("use True or False as inputs to --clean") 
    
    if interpolate=='True':
        interpolate=True
        print("WILL INTERPOLATE")
    elif interpolate=='False':
        interpolate=False
        print("WILL NOT INTERPOLATE")

    else:
        raise AssertionError("use True or False as inputs to --interpolate") 
    
    if off_pointing=='True':
        off_pointing=True
    elif off_pointing=='False':
        off_pointing=False
    else:
        raise AssertionError("use True or False as inputs to --off_pointing") 
    
    
    if clean:
        print("WILL CLEAN")
        tag='rfi_cleaned'
    else:
        print("WILL NOT CLEAN")
        tag='unclean'
    if off_pointing:
        logging.info("Using BB position")
        tag+='off_baseband_pointing'
    database=kko_events_database

    df_pulsars_loc=pd.read_csv('/arc/home/shiona/archive/old_plot_data/pulsar_localization_results.csv')
    df_pulsars_loc=df_pulsars_loc.sort_values(by='errors',ascending=False).reset_index(drop=True)
    df_cal=get_calibrator_dataframe()
    
    if event_id==0:
        events=np.array(df_pulsars_loc['event_id'])[nstart:nstop]
    else:
        events=[event_id]#307063854,306336576,304499017,268914678]
                    # 307063854= B0136+57, 74.0004882812
       
    print(f"WILL PROCESS EVENTS {events}")
    if test==1:
        assert True==False
    log_folder='/arc/home/shiona/'
    if event_id==0:
        log_file_event = os.path.join(log_folder, f'{nstart}_to_{nstop}_{events[0]}_to_{events[-1]}_{tag}_beamform_and_correlate.log')
    else:
        log_file_event = os.path.join(log_folder, f'{event_id}_{tag}_beamform_and_correlate.log')

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        handlers=[logging.FileHandler(log_file_event),  # Output log messages to a log file specific to this event
        logging.StreamHandler()],  # Output log messages to the console
        force=True
    )
    logging.info(f"writing to {log_file_event}")
    print(events)
    for event_id in events:
        event = get_event_data(event_id,events_database=database)
        DM=event['DM'][0]
        src_name = event["source_name"][0]
        year = str(event["year"][0])
        month = int(event["month"][0])
        day = int(event["day"][0])
        ctime=float(event['ctime'][0])
        try:
            df_pulsars_locx=df_pulsars_loc[df_pulsars_loc['event_id']==event_id].reset_index(drop=True)
            cal_name=df_pulsars_locx['cal_name'][0]
        except:
            df_pulsars_locx=df_pulsars_loc[df_pulsars_loc['target']==src_name].reset_index(drop=True)
            cal_name=df_pulsars_locx['cal_name'][0]
        if off_pointing:
            logging.info("Using BB position")
            from outriggers_vlbi_pipeline.query_database import find_files, get_baseband_localization_info
            try:
                bbinfo=get_baseband_localization_info(event_id)
                ra=bbinfo['ra']
                dec=bbinfo['dec']
                ras=[ra]
                decs=[dec]
            except:
                df_pointings=pd.read_csv('/arc/home/shiona/100_pulsar_pointings.csv')
                df_pointingsx=df_pointings[df_pointings['event_id']==event_id].reset_index(drop=True)
                ra=df_pointingsx['ra'][0]
                dec=df_pointingsx['dec'][0]
                ras=[ra]
                decs=[dec]
        else:#except:
            logging.info("Using TRUE position")
            ratrue,dectrue=get_true_pulsar_pos(src_name,ctime=ctime)
            ras=[ratrue]#+np.random.uniform(-1/60,1/60)]
            decs=[dectrue]#+np.random.uniform(-1/60,1/60)]
        src_names=[src_name]

        cal_match=df_cal[df_cal['name']==cal_name].reset_index(drop=True)
        src_names.append(cal_name)
        ras.append(cal_match['ra_j2000'][0])
        decs.append(cal_match['dec_j2000'][0])

        src_names=np.array(src_names)
        ras=np.array(ras)
        decs=np.array(decs)
        print(src_names)
        
        cal_files = find_files(
            event_id, data_type="visibilities", source_type="calibrator",version=version,filename_suffix=cal_name+'*'+tag
        )
        target_files = find_files(
            event_id, data_type="visibilities", source_type="target",version=version,filename_suffix=tag
        )        
        print(target_files)
        if False: #len(cal_files)>0 and len(target_files)>0:
            logging.info(f"##############################################")
            logging.info(f"{event_id} has already been processed!")
            logging.info(f"##############################################")
        else:
            try:
                #telescopes=[kko,chime]
                tel_names=['chime','kko']
                logging.info(f'EVENT INFORMATION: {event_id}, {src_name}')

                data_pulled_tels=[]
                raw_data_dirs=[]
                for telescope in tel_names:
                    data_pulled_tel=False
                    data_exists,raw_data_dir=baseband_exists(event_id, telescope)
                    if data_exists==False:
                        datatrail_pull_or_clear(event_id, telescope=telescope, cmd_str='PULLED')
                        data_pulled_tel=True
                        data_exists,raw_data_dir=baseband_exists(event_id, telescope)
                    assert data_exists==True, f"data for telescope {telescope} not found"
                    logging.info(f'Baseband data for {telescope} found on /arc. Moving on to beamforming stage...')
                    ### BEAMFORM ###
                    data_pulled_tels.append(data_pulled_tel)
                    raw_data_dirs.append(raw_data_dir)
                    proceed=True

            except:
                logging.info(f"{event_id} FAILED TO PROCESS")
                proceed=False
                
            if proceed:
                for m in range(len(tel_names)):
                    telescope=tel_names[m]
                    data_pulled=data_pulled_tels[m]
                    raw_data_dir=raw_data_dirs[m]

                    cal_h5 = get_calfiles(day=day,month=month,year=year,telescope=telescope, ctime=ctime)
                    if interpolate:#and telescope=='kko':
                        if day<10:
                            day_str='0'+str(day)
                        else:
                            day_str=str(day)
                        if month<10:
                            month_str='0'+str(month)
                        else:
                            month_str=str(month)

                        files=glob.glob(f'/arc/home/shiona/{telescope}_interpolated_gains/gain_{year}{month_str}{day_str}*.h5')
                        if len(files)==0:
                            from KLT_filter.interpolate_gains import make_interpolated_gain_file
                            cal_h5=make_interpolated_gain_file(cal_h5,telescope=telescope)
                        else:
                            cal_h5=files[0]#[0]
                        print(f"USING INTERPOLATED GAINS {cal_h5}")
                    else:
                        print("WILL NOT USE INTERPOLATED GAINS")
                    print("STARTING BEAMFORMING")

                    if True:
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
                            source_types=['target','calibrator'],
                            tag=tag,
                            clean=clean,
                        )

                                                    
                    if telescope=='chime':
                        chime_target_singlebeam=out_files[0]
                        chime_cal_singlebeam=out_files[1]
                    if telescope=='kko':
                        kko_target_singlebeam=out_files[0]
                        kko_cal_singlebeam=out_files[1]

                    if data_pulled and del_data:
                        datatrail_pull_or_clear(event_id, telescope=telescope, cmd_str='CLEARED')


                diagnostics_out_dir=get_full_filepath(event_id=event_id, data_type="diagnostics",events_database=kko_events_database)

                ## target first ##
                source_type='target'
                get_pulse_lims=True
                tel_singlebeams=[chime_target_singlebeam,kko_target_singlebeam]
                logging.info(f"Using {tel_singlebeams}")
                vis_target=re_correlate_target(
                    event_id,DM=DM,source_type=source_type,telescopes=[chime,kko],
                    tel_singlebeams=tel_singlebeams,get_pulse_lims=get_pulse_lims,
                    diagnostics_out_dir=diagnostics_out_dir,save_autos=True)

                vis_dir = get_full_filepath(event_id=event_id, data_type="visibilities",source_type=source_type,events_database=kko_events_database)
                source_name=vis_target['index_map']['pointing_center']['source_name'][0].astype(str)
                vis_out_file_target = f"{vis_dir}{event_id}_{source_name}_{tag}_vis.h5"
                os.makedirs(os.path.dirname(vis_out_file_target), exist_ok=True, mode=0o777)
                logging.info(f"Saving visibilities to {vis_out_file_target}")
                vis_target.save(vis_out_file_target)


                ## cal second ##
                source_type='calibrator'
                get_pulse_lims=False
                tel_singlebeams=[chime_cal_singlebeam,kko_cal_singlebeam]
                logging.info(f"Using {tel_singlebeams}")
                vis_target=re_correlate_target(
                    event_id,DM=0,source_type=source_type,telescopes=[chime,kko],
                    tel_singlebeams=tel_singlebeams,get_pulse_lims=get_pulse_lims,
                    diagnostics_out_dir=diagnostics_out_dir,save_autos=True)

                vis_dir = get_full_filepath(event_id=event_id, data_type="visibilities",source_type=source_type,events_database=kko_events_database)
                source_name=vis_target['index_map']['pointing_center']['source_name'][0].astype(str)
                vis_out_file_target = f"{vis_dir}{event_id}_{source_name}_{tag}_vis.h5"
                os.makedirs(os.path.dirname(vis_out_file_target), exist_ok=True, mode=0o777)
                logging.info(f"Saving visibilities to {vis_out_file_target}")
                vis_target.save(vis_out_file_target)



