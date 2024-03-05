from outriggers_vlbi_pipeline.vlbi_pipeline_config import chime,kko,gbo
import sys
import numpy as np
from glob import glob

from rfi_shion_version import clean_persistent_rfi

import baseband_analysis.core.calibration as cal

from outriggers_vlbi_pipeline.known_calibrators import get_true_pulsar_pos
from ch_util import tools
import re
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
from outriggers_vlbi_pipeline.arc_commands import datatrail_pull_or_clear, datatrail_pull_cmd,datatrail_clear_cmd,baseband_exists,delete_baseband,vchmod,delete_multibeam,data_exists_at_minoc,datatrail_pull,datatrail_clear




def freq_id_from_filename(
    filename: str
) -> int:
    """Gets frequency id from filename of raw baseband data; in principle faster but perhaps less reliable than just extracting the header information from the .h5 data.
    Assumes that rawdata is written in the form baseband_<EVENT_ID>_FREQ_ID.h5

    Inputs: 
    ----------
    filename: str
        Name of raw baseband file
    Returns: 
    ----------
    frequency id corresponding corresponding to raw beamformed data
    """
    
    try:
        freq_id = re.split("_|.h5", filename)[-2]
        return int(freq_id)
    except:
        raise AssertionError(
            "raw data is not of the form baseband_<EVENT_ID>_FREQ_ID.h5."
        )
        
        
event_id=268914678



#ra_cal=140.400962818458
#dec_cal=62.2644945419444

ra_cal=24.2441450135833
dec_cal=47.8580833533333
cal_name='J013658.5+475129'


ra_cal2=24.832133450058727
dec_cal2=33.15974373333333
cal_name2='J0137+3309'


#cal='J013658.5+475129'
#target='J0137+3309'


clean=True


#telescopes=['chime']

events=[308546393,308328717, 308294368, 308524547, 308863040, 309041048, 308263714,
       308557728, 308505094, 308566243, 307700708, 308355286, 309032189,
       307883964, 309077224, 309080209, 308807047, 309060071, 308787399,
       308766790, 311595774, 309097039, 308835395, 307698645, 308803721,
       308816441, 306481006, 305612465, 306818801, 306327369, 304592723,
       306497548, 302615544, 302267093, 302202883, 302043210, 302017525,
       301891801, 301718513, 301531256, 301505535, 301310468, 301083954,
       301050157, 300800731, 300770882, 300520925, 299930493, 294112275,
       295115852, 299441097, 299723484, 299404307, 299253234, 298159260]
'''350429056,
350429283,
350429530,
350429757,
350430021,350430272,350430491,350431321,'''
events=[350431671,350431908,350432181][1:]

#events=[20240125012114,20240125012138]
t
elescopes=['kko','chime'] #'gbo',
stations=[chime,kko]#,gbo]

#events=[350432181]
if False: #for event_id in events:
    event = get_event_data(event_id,events_database=kko_events_database,version='0.2test')
    src_name = event["source_name"][0]
    year = str(event["year"][0])
    month = int(event["month"][0])
    day = int(event["day"][0])
    ctime=float(event['ctime'][0])
    ratrue,dectrue=get_true_pulsar_pos(src_name,ctime=ctime)
    ra_src=ratrue
    dec_src=dectrue

    decs=[dectrue,dec_cal,dec_cal2]
    ras=[ratrue,ra_cal,ra_cal2]
    src_names=[src_name,cal_name,cal_name2]
    ras=np.array(ras)
    decs=np.array(decs)

    
    for telescope in telescopes:
        for source in range(2,len(ras)):
            raw_data_dir=f'/arc/projects/chime_frb/temp/data/{telescope}/baseband/raw/*/*/*/astro_{event_id}/*.h5'
            print(raw_data_dir)
            datafiles_all=glob(raw_data_dir)
            
            if len(datafiles_all)==0:
                raw_data_dir=f'/arc/projects/chime_frb/vlbi/chime_rawdata/astro_{event_id}/*.h5'
                datafiles_all=glob(raw_data_dir)
                
                if len(datafiles_all)==0:
                    datatrail_pull_or_clear(event_id, telescope=telescope, cmd_str='PULLED')
                    raw_data_dir=f'/arc/projects/chime_frb/temp/data/{telescope}/baseband/raw/*/*/*/astro_{event_id}/*.h5'
                    print(raw_data_dir)
                    datafiles_all=glob(raw_data_dir)

            datafiles_all.sort(key=freq_id_from_filename)
            print(datafiles_all[0])
            first_data = bbdata.BBData.from_acq_h5(datafiles_all[0])
            ctime = first_data["time0"]["ctime"][0]
            ratrue,dectrue=get_true_pulsar_pos(src_name,ctime=ctime)
            ra_src=ratrue
            dec_src=dectrue

            decs=[dectrue,dec_cal,dec_cal2]
            ras=[ratrue,ra_cal,ra_cal2]
            src_names=[src_name,cal_name,cal_name2]
            ras=np.array(ras)
            decs=np.array(decs)

            if telescope=='kko':
                backend=kko_backend
                reference_feed = tools.ArrayAntenna(
                        id=-1, slot=-1, powered=True, flag=True, pos=[0, 0, 0], delay=0
                ) 
                static_delays=False
                cal_h5 = get_calfiles(day=day,month=month,year=year,telescope=telescope, ctime=ctime)

            elif telescope=='gbo':
                backend=gbo_backend
                reference_feed = tools.ArrayAntenna(
                        id=-1, slot=-1, powered=True, flag=True, pos=[0, 0, 0], delay=0
                ) 
                static_delays=False
                cal_h5 = get_calfiles(day=day,month=month,year=year,telescope=telescope, ctime=ctime,gain_calibrator='casa')

            else:
                backend=chime_backend
                reference_feed=tools.CHIMEAntenna(
                        id=-1, slot=-1, powered=True, flag=True, pos=[0, 0, 0]
                )
                static_delays = backend.static_delays

                cal_h5 = get_calfiles(day=day,month=month,year=year,telescope=telescope, ctime=ctime)


            import os
            output_file_dir=f'/arc/projects/chime_frb/shiona/rfi_analysis/{event_id}'
            os.umask(0)
            os.makedirs(
                output_file_dir, exist_ok=True, mode=0o777
            )  # if output directory doesn't exist, create it
            output_file_dir=f'/arc/projects/chime_frb/shiona/rfi_analysis/{event_id}/singlebeams/'
            os.umask(0)
            os.makedirs(
                output_file_dir, exist_ok=True, mode=0o777
            )  # if output directory doesn't exist, create it


            if clean:
                out_file=f'{output_file_dir}{event_id}_{telescope}_singlebeam_{src_names[source]}_clean.h5'
            else:
                out_file=f'{output_file_dir}{event_id}_{telescope}_singlebeam_{src_names[source]}.h5'
            print(out_file)


            datas=[]
            date = datetime.utcfromtimestamp(ctime)
            chunk_size=8
            for i in range(len(datafiles_all)//chunk_size):
                print(f"{i*chunk_size}-{i*chunk_size+chunk_size} of {len(datafiles_all)} for telescope {telescope}!")
                files_to_process=datafiles_all[i*chunk_size:i*chunk_size+chunk_size]

                try:
                    data = bbdata.BBData.from_acq_h5(files_to_process)
                    inputs = tools.get_correlator_inputs(date, correlator=backend.correlator)
                    gains = cal.read_gains(cal_h5)
                    cal.apply_calibration(data, gains, inputs=inputs)
                    print([ras[source]])
                    print([decs[source]])
                    if clean:
                        clean_persistent_rfi(
                            data=data, ra=np.array([ras[source]]), dec=np.array([decs[source]]), ps_gain_file=cal_h5,inputs=inputs,
                            reference_feed=reference_feed,static_delays=static_delays,
                            obs=backend.obs)

                    beamform.tied_array(
                            data,
                            ra=np.array([ras[source]]),
                            dec=np.array([decs[source]]),
                            source_name=np.array([src_names[source]]),
                            correlator_inputs=inputs,
                            obs=backend.obs,
                            reference_feed=reference_feed,
                            static_delays=static_delays,#telescope_rotation=telescope_rot
                        )

                    del data["baseband"]
                    datas.append(data)

                except:
                    for j in range(len(files_to_process)):#datafiles_all[i*chunk_size:i*chunk_size+chunk_size])):
                        try:
                            file=files_to_process[j]

                            data = bbdata.BBData.from_acq_h5(file)

                            inputs = tools.get_correlator_inputs(date, correlator=backend.correlator)
                            gains = cal.read_gains(cal_h5)
                            cal.apply_calibration(data, gains, inputs=inputs)

                            print([ras[source]])
                            print([decs[source]])
                            if clean:
                                clean_persistent_rfi(
                                    data=data, ra=np.array([ras[source]]), dec=np.array([decs[source]]), ps_gain_file=cal_h5,inputs=inputs,
                                    reference_feed=reference_feed,static_delays=static_delays,
                                    obs=backend.obs)

                            beamform.tied_array(
                                    data,
                                    ra=np.array([ras[source]]),
                                    dec=np.array([decs[source]]),
                                    source_name=np.array([src_names[source]]),
                                    correlator_inputs=inputs,
                                    obs=backend.obs,
                                    reference_feed=reference_feed,
                                    static_delays=static_delays,#telescope_rotation=telescope_rot
                                )

                            del data["baseband"]
                            datas.append(data)
                        except:
                            print(f"could not process {file}")



            beamformed_rfi_cleaned = bbdata.concatenate(datas)
            print(out_file)
            beamformed_rfi_cleaned.save(out_file)
            beamformed_rfi_cleaned.close()
            datatrail_pull_or_clear(event_id, telescope=telescope, cmd_str='CLEARED')
            

    
source_types=['target','calibrator','calibrator']
get_autolims=[True,False,False]
from outriggers_vlbi_pipeline.cross_correlate_data import cross_correlate_data    
for event_id in events:
    file_dir=f'/arc/projects/chime_frb/shiona/rfi_analysis/{event_id}/singlebeams/'
    event = get_event_data(event_id,events_database=kko_events_database,version='0.2test')
    src_name = event["source_name"][0]
    DM=event['DM'][0]
    src_names=[src_name,cal_name,cal_name2]
    for i in range(2,len(src_names)):
        src=src_names[i]
        tel_names=[tel.info.name for tel in stations]
        tel_bbdatas=[]
        for telname in tel_names:
            file=glob(f'{file_dir}*{telname}*{src}*clean*')[0]
            data=BBData.from_file(file)
            tel_bbdatas.append(data)
        cross_correlate_data(
                event_id,
                tel_bbdatas,
                stations,
                source_name=src,
                source_type=source_types[i],
                DM=DM,
                get_autolims=get_autolims[i],
                save_beamformed=True, ###TODO: this should be saved outside of cross correlate function, cross correlate function should be called by recorrelate functions!!
            )
        
        
    