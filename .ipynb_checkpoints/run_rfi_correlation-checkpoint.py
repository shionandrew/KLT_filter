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
from clean_correlation import clean_and_form_singlebeams
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
    parser.add_argument("--group", help="Group to process.", default=0, type=int)
    parser.add_argument("--event_id", help="Version of pipeline", type=int)
    cmdargs = parser.parse_args()
    group=cmdargs.group
    logging.info(group)
    database=kko_events_database
    Nchunk=20
    df_pulsars_loc=pd.read_csv('/arc/home/shiona/pulsar_localization_results.csv')
    df_pulsars_loc=df_pulsars_loc.sort_values(by='errors',ascending=False).reset_index(drop=True)
    df=get_calibrator_dataframe()
    events=[322017258, 320930172, 316959442, 314715728, 314006842, 314210443,
           313966747, 313501776, 311695186, 311744688, 311778391, 309991107,
           310032717, 309097039, 309080209, 309032189, 309077224, 308807047,
           309060071, 308787399, 308863040, 308835395, 308816441, 308766790,
           308546393, 308524547, 308566243, 308557728, 308328717, 308294368,
           308355286, 308026938, 307063854, 306481006, 306497548, 306336576,
           305661586, 304592723, 304499017, 304522001, 304116324, 304050301,
           273590077, 268348878, 255950332, 255995450, 255860695, 255781812,
           255670378, 254860456, 254387820, 254019109, 253853758, 253713145,
           253218314]
    print(df.keys())
    clean=False
    print(events[group*Nchunk:group*Nchunk+Nchunk])
    tag='_off_pointing_not_clean'
    #df_pulsars_loc=df_pulsars_loc[df_pulsars_loc['event_id']==312385630].reset_index(drop=True)
    for event_id in events[group*Nchunk:group*Nchunk+Nchunk]:#i in range(len(df_pulsars_loc))[group*Nchunk:group*Nchunk+Nchunk]:
        df=get_calibrator_dataframe()
        #event_id=df_pulsars_loc['event_id'][i]
        df_pulsars_locx=df_pulsars_loc[df_pulsars_loc['event_id']==event_id].reset_index(drop=True)
        cal_name=df_pulsars_locx['cal_name'][0]
        event = get_event_data(event_id,events_database=database)
        DM=event['DM'][0]
        src_name = event["source_name"][0]
        year = str(event["year"][0])
        month = int(event["month"][0])
        day = int(event["day"][0])
        ctime=float(event['ctime'][0])
        try:
            from outriggers_vlbi_pipeline.query_database import find_files, get_baseband_localization_info
            bbinfo=get_baseband_localization_info(event_id)
            ra=bbinfo['ra']
            dec=bbinfo['dec']
            ras=[ra]
            decs=[dec]
        except:
            tag+='_no_bb'
            target_file=find_files(event_id,data_type='visibilities',source_type='target',filename_suffix='off_pointing')[0]
            vis=VLBIVis.from_file(target_file)
            ra=vis.index_map['pointing_center']['corr_ra'][0]
            dec=vis.index_map['pointing_center']['corr_dec'][0]
            ras=[ra]
            decs=[dec]
            #ratrue,dectrue=get_true_pulsar_pos(src_name,ctime=ctime)
            #ras=[ratrue+np.random.uniform(-1/60,1/60)]
            #decs=[dectrue+np.random.uniform(-1/60,1/60)]
        src_names=[src_name]

        cal_match=df[df['name']==cal_name].reset_index(drop=True)
        src_names.append(cal_name)
        ras.append(cal_match['ra_j2000'][0])
        decs.append(cal_match['dec_j2000'][0])

        src_names=np.array(src_names)
        ras=np.array(ras)
        decs=np.array(decs)

        log_folder=get_full_filepath(event_id=event_id,data_type='logs',events_database=database)
        os.makedirs(log_folder, exist_ok=True)
        log_file_event = os.path.join(log_folder, f'{event_id}_beamform_and_correlate.log')
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
            handlers=[logging.FileHandler(log_file_event),  # Output log messages to a log file specific to this event
            logging.StreamHandler()],  # Output log messages to the console
            force=True
        )
        logging.info(f"writing to {log_file_event}")
        complete=check_correlation_completion(event_id,version=version)
        cal_files = find_files(
            event_id, data_type="visibilities", source_type="calibrator",version=version,filename_suffix=cal_name+'*'+tag
        )
        target_files = find_files(
            event_id, data_type="visibilities", source_type="target",version=version,filename_suffix=tag
        )        
        if len(cal_files)>0 and len(target_files)>0:
            logging.info(f"##############################################")
            logging.info(f"{event_id} has already been processed!")
            logging.info(f"##############################################")
            proceed=False

        else:
            telescopes=['kko','chime']
            ### BASIC EVENT INFO + PULL RAW BASEBAND ###
            event = get_event_data(event_id,events_database=database)
            src_name = event['source_name'][0]
            logging.info(f'EVENT INFORMATION: {event_id}, {src_name}')
            proceed=True
        if proceed and event_id!=308984712:
            if proceed:
                for telescope in telescopes:
                    data_pulled=False
                    chime_singlebeam=find_files(event_id,data_type='singlebeams',source_type='calibrator',telescope='chime')
                    kko_singlebeam=find_files(event_id,data_type='singlebeams',source_type='calibrator',telescope='kko')

                    if len(chime_singlebeam)>0 and len(kko_singlebeam)>0:
                        logging.info(f"beamformed files for {telescope} already found! moving to next stage...")
                    
                    target_files=find_files(event_id=event_id,data_type="singlebeams",telescope='*',source_type='target',filename_suffix=telescope+'*'+tag)
                    if len(target_files)>1:
                        proceed=True
                    elif proceed==True: #do not continue if there was no datra at kko
                        data_exists,raw_data_dir=baseband_exists(event_id, telescope)
                        if data_exists==False:
                            at_minoc=data_exists_at_minoc(event_id, telescope)
                            if at_minoc:
                                datatrail_pull_or_clear(event_id, telescope=telescope, cmd_str='PULLED')
                                data_pulled=True
                                data_exists,raw_data_dir=baseband_exists(event_id, telescope)
                        if data_exists==True:
                            logging.info(f'Baseband data for {telescope} found on /arc. Moving on to beamforming stage...')
                            ### BEAMFORM ###
                            cal_h5 = get_calfiles(day=day,month=month,year=year,telescope=telescope, ctime=ctime)

                            clean_and_form_singlebeams(
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

                            if data_pulled:
                                datatrail_pull_or_clear(event_id, telescope=telescope, cmd_str='CLEARED')

                        else:
                            proceed=False
                            logging.error(f'Baseband data for {telescope} NOT found!')
                            update_event_status(event_id,'error',f'Baseband data for {telescope} NOT found!',events_database=database)

                    target_files=find_files(event_id=event_id,data_type="singlebeams",telescope='*',source_type='target',filename_suffix=telescope+'*'+tag)
                    if len(target_files)<1:
                        proceed=False
                        logging.error(f'Unable to correlate data for event id {event_id}!')
                    else:
                        proceed=True
                if proceed:
                    ### CROSS CORRELATE TARGET DATA ###
                    diagnostics_out_dir=get_full_filepath(event_id=event_id, data_type="diagnostics",events_database=kko_events_database)
                    vis_target=re_correlate_target(event_id,DM=DM,telescopes=[chime,kko],diagnostics_out_dir=diagnostics_out_dir)
                    vis_dir = get_full_filepath(event_id=event_id, data_type="visibilities",source_type='target',events_database=kko_events_database)
                    source_name=vis_target['index_map']['pointing_center']['source_name'][0].astype(str)
                    vis_out_file = f"{vis_dir}{event_id}_{source_name}_finalbaseline{tag}_vis.h5"
                    os.makedirs(os.path.dirname(vis_out_file), exist_ok=True, mode=0o777)
                    logging.info(f"Saving visibilities to {vis_out_file}")
                    vis_target.save(vis_out_file)

                    ### CROSS CORRELATE CAL DATA ###
                    tel_singlebeams=[]
                    chime_singlebeam=find_files(event_id,data_type='singlebeams',source_type='calibrator',
                                            filename_suffix='chime*'+cal_name+f'*{tag}*')[0]
                    tel_singlebeams.append(chime_singlebeam)
                    kko_singlebeam=find_files(event_id,data_type='singlebeams',source_type='calibrator',
                                            filename_suffix='kko*'+cal_name+f'*{tag}*')[0]
                    tel_singlebeams.append(kko_singlebeam)
                    logging.info(tel_singlebeams)
                    vis_calibrator=re_correlate_target(
                        event_id,DM=DM,source_type='calibrator',
                        source_name=cal_name,telescopes=[chime,kko],
                        tel_singlebeams=tel_singlebeams,get_pulse_lims=False,
                        diagnostics_out_dir=diagnostics_out_dir)
                    source_name=vis_calibrator['index_map']['pointing_center']['source_name'][0].astype(str)
                    vis_dir = get_full_filepath(event_id=event_id, data_type="visibilities",
                                                source_type='calibrator',events_database=kko_events_database)
                    vis_out_file = f"{vis_dir}{event_id}_{source_name}_finalbaseline{tag}_vis.h5"
                    os.makedirs(os.path.dirname(vis_out_file), exist_ok=True, mode=0o777)
                    logging.info(f"Saving visibilities to {vis_out_file}")
                    vis_calibrator.save(vis_out_file)

