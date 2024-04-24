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
from outriggers_vlbi_pipeline.cross_correlate_data import re_correlate_target,_cross_correlate_data

from outriggers_vlbi_pipeline.calibration import get_calibrator_visibilities, make_calibrated_visibilities
from KLT_filter.clean_correlation import clean_and_form_singlebeams
from baseband_analysis.core import BBData
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
    events_database=kko_events_database
    df_pulsars_loc=pd.read_csv('/arc/home/shiona/archive/old_plot_data/pulsar_localization_results.csv')
    print("XXX")
    #rfi cleaned 304499017,307063854,304997126,306481006,268914678
    #tag='unclean'#'rfi_cleaned'#'unclean'  #unclean 304499017,304116324,268914678
    tag='unclean'
    '''308816441, #304499017
 308863040,
 308807047,
 308766790,
 314348840,
 309097039,
 310032717,
 313966747,
 312496248,
 314210443,
 314004518,
 311778391,
 311744688,
 311695186,
 311653530,
 311637793,
 307608701,
 308294368,
 307063854,268348878,
 282778989,
 304409158,
 304050301,
 305661586, 306497548,
 320120393,
 320047467,
 320930172,
 319938851,
 314715728,
 273590077,
 268914678, 270754345,304116324,304499017,'''
    events=[350258229,350258482,350258834]#350256958,350257224,350257489,350257811,350258229,350258482,350258834,350259263,350259503][0:2]
    for event_id in events:
        tags=['','_clean'][1:]
        tag2s=['unclean','clean']
        for i in range(len(tags)):
            tag=tags[i]
            tag2=tag2s[i]
            
            telescopes=[chime,gbo]

            DM=74.0004882812
            src_name='B0136+57'
            chime_file=f'/arc/projects/chime_frb/shiona/rfi_analysis/{event_id}/singlebeams/{event_id}_chime_singlebeam_{src_name}{tag}.h5'
            #out_file=f'/arc/projects/chime_frb/shiona/rfi_analysis/{event_id}/singlebeams/{event_id}_gbo_singlebeam_{src_name}{tag}.h5'
            out_file=f'/arc/projects/chime_frb/vlbi/revrfi_test2/2024/01/07/{event_id}/target_singlebeams/{event_id}_gbo_singlebeam_{src_name}{tag2}.h5'
            
            print(out_file)
            print(chime_file)
            telescopes=[chime,gbo]
            source_type='target'
            get_autolims=True
            tel_bbdatas=[BBData.from_file(chime_file),BBData.from_file(out_file)]
            vis_target = _cross_correlate_data(
                event_id=event_id,
                tel_bbdatas=tel_bbdatas,
                telescopes=telescopes,
                source_name=src_name,
                source_type=source_type,
                get_autolims=get_autolims,
                DM=DM,
                events_database=events_database,
                save_autos=True,
                )
            save_file=f'/arc/projects/chime_frb/shiona/rfi_analysis/{event_id}/visibilities/{event_id}_chime_gbo_visibilities_{src_name}{tag}.h5'
            os.makedirs(os.path.dirname(save_file), exist_ok=True, mode=0o777)
            print(save_file)
            vis_target.save(save_file)
            
            src_name='J013658.5+475129'
            chime_file=f'/arc/projects/chime_frb/shiona/rfi_analysis/{event_id}/singlebeams/{event_id}_chime_singlebeam_{src_name}{tag}.h5'
            #out_file=f'/arc/projects/chime_frb/shiona/rfi_analysis/{event_id}/singlebeams/{event_id}_gbo_singlebeam_{src_name}{tag}.h5'
            out_file=f'/arc/projects/chime_frb/vlbi/revrfi_test2/2024/01/07/{event_id}/calibrator_singlebeams/{event_id}_gbo_singlebeam_{src_name}{tag2}.h5'
            print(out_file)
            print(chime_file)
            source_type='calibrator'
            get_autolims=False
            tel_bbdatas=[BBData.from_file(chime_file),BBData.from_file(out_file)]
            vis_J013658 = _cross_correlate_data(
                event_id=event_id,
                tel_bbdatas=tel_bbdatas,
                telescopes=telescopes,
                source_name=src_name,
                source_type=source_type,
                get_autolims=get_autolims,
                DM=0,
                events_database=events_database,
                save_autos=True,
                )
            save_file=f'/arc/projects/chime_frb/shiona/rfi_analysis/{event_id}/visibilities/{event_id}_chime_gbo_visibilities_{src_name}{tag}.h5'
            os.makedirs(os.path.dirname(save_file), exist_ok=True, mode=0o777)
            print(save_file)
            vis_J013658.save(save_file)
            
            
            src_name='J0137+3309'
            chime_file=f'/arc/projects/chime_frb/shiona/rfi_analysis/{event_id}/singlebeams/{event_id}_chime_singlebeam_{src_name}{tag}.h5'
            #out_file=f'/arc/projects/chime_frb/shiona/rfi_analysis/{event_id}/singlebeams/{event_id}_gbo_singlebeam_{src_name}{tag}.h5'
            out_file=f'/arc/projects/chime_frb/vlbi/revrfi_test2/2024/01/07/{event_id}/calibrator_singlebeams/{event_id}_gbo_singlebeam_{src_name}{tag2}.h5'

            print(out_file)
            print(chime_file)
            source_type='calibrator'
            get_autolims=False
            tel_bbdatas=[BBData.from_file(chime_file),BBData.from_file(out_file)]
            vis_J0137 = _cross_correlate_data(
                event_id=event_id,
                tel_bbdatas=tel_bbdatas,
                telescopes=telescopes,
                source_name=src_name,
                source_type=source_type,
                get_autolims=get_autolims,
                DM=0,
                events_database=events_database,
                save_autos=True,
                )
            save_file=f'/arc/projects/chime_frb/shiona/rfi_analysis/{event_id}/visibilities/{event_id}_chime_gbo_visibilities_{src_name}{tag}.h5'
            os.makedirs(os.path.dirname(save_file), exist_ok=True, mode=0o777)
            print(save_file)
            vis_J0137.save(save_file)

        
        
        
        '''try:
            df_pulsars_locx=df_pulsars_loc[df_pulsars_loc['event_id']==event_id].reset_index(drop=True)
            cal_name=df_pulsars_locx['cal_name'][0]
        except:
            df_pulsars_locx=df_pulsars_loc[df_pulsars_loc['target']==src_name].reset_index(drop=True)
            cal_name=df_pulsars_locx['cal_name'][0]        

        vis_cal=re_correlate_target(
            event_id,DM=0,source_type=source_type,telescopes=[chime,kko],
            tel_singlebeams=tel_singlebeams,get_pulse_lims=get_pulse_lims,
            save_autos=True)'''
        

        '''## cal second ##
        source_type='calibrator'
        get_pulse_lims=False
        kko_cal_singlebeam=''
        singlebeams=find_files(event_id,source_type=source_type,data_type='singlebeams',version='rfi_test2')#,telescope='kko')
        for file in singlebeams:
            if 'kko' in file:
                if tag in file and cal_name in file:
                    kko_cal_singlebeam=file   
        if kko_cal_singlebeam=='':
            singlebeams=find_files(event_id,source_type=source_type,data_type='singlebeams',version='rfi_test')#,telescope='kko')
            for file in singlebeams:
                if 'kko' in file:
                    if tag in file and cal_name in file:
                        kko_cal_singlebeam=file  
        chime_cal_singlebeam=''
        for file in singlebeams:
            if 'chime' in file and 'kko' not in file:
                if tag in file and cal_name in file:
                    chime_cal_singlebeam=file

        tel_singlebeams=[chime_cal_singlebeam,kko_cal_singlebeam]
        print(f"Using {tel_singlebeams}")
        vis_cal=re_correlate_target(
            event_id,DM=0,source_type=source_type,telescopes=[chime,kko],
            tel_singlebeams=tel_singlebeams,get_pulse_lims=get_pulse_lims,
            save_autos=True)
        

        vis_dir = get_full_filepath(event_id=event_id, data_type="visibilities",source_type=source_type,events_database=kko_events_database)
        source_name=vis_cal['index_map']['pointing_center']['source_name'][0].astype(str)
        vis_out_file_cal = f"{vis_dir}{event_id}_{source_name}_{tag}_vis.h5"
        os.makedirs(os.path.dirname(vis_out_file_cal), exist_ok=True, mode=0o777)
        print(f"Saving visibilities to {vis_out_file_cal}")
        vis_cal.save(vis_out_file_cal)'''
        '''
        ## target ##
        source_type='target'
        get_pulse_lims=True
        kko_cal_singlebeam=''
        singlebeams=find_files(event_id,source_type=source_type,data_type='singlebeams',version='rfi_test2')#,telescope='kko')
        for file in singlebeams:
            if 'kko' in file:
                if tag in file:
                    kko_cal_singlebeam=file    
        chime_cal_singlebeam=''
        for file in singlebeams:
            if 'chime' in file and 'kko' not in file:
                if tag in file:
                    chime_cal_singlebeam=file

        tel_singlebeams=[chime_cal_singlebeam,kko_cal_singlebeam]
        print(f"Using {tel_singlebeams}")
        vis_target=re_correlate_target(
            event_id,DM=DM,source_type=source_type,telescopes=[chime,kko],
            tel_singlebeams=tel_singlebeams,get_pulse_lims=get_pulse_lims,plot_diagnostics=False,
            save_autos=True)
        

        vis_dir = get_full_filepath(event_id=event_id, data_type="visibilities",source_type=source_type,events_database=kko_events_database)
        source_name=vis_target['index_map']['pointing_center']['source_name'][0].astype(str)
        vis_out_file_target = f"{vis_dir}{event_id}_{source_name}_{tag}_vis.h5"
        os.makedirs(os.path.dirname(vis_out_file_target), exist_ok=True, mode=0o777)
        print(f"Saving visibilities to {vis_out_file_target}")
        vis_target.save(vis_out_file_target)'''

        '''vis_out_file_target=glob.glob(f'/arc/projects/chime_frb/vlbi/revrfi_test2/*/*/*/{event_id}/target_visibilities/*{tag}*')[0]
        vis_out_file_cal=glob.glob(f'/arc/projects/chime_frb/vlbi/revrfi_test2/*/*/*/{event_id}/calibrator_visibilities/*{tag}*')[0]
        out_tag=tag+'off_pointing'
        print(event_id)
        print(vis_out_file_target)
        print(vis_out_file_cal)
        make_calibrated_visibilities(event_id=event_id,target_file=vis_out_file_target,cal_files=[vis_out_file_cal],out_tag=out_tag,write=True,debug=True,overwrite=True)'''



