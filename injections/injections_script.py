import sys
import numpy as np
from glob import glob

from injection_rfi_shion_version import clean_persistent_rfi

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
from simulate_signal import convert_inputs_to_stations
import astropy.units as un

import astropy.coordinates as ac

chime = ac.EarthLocation.from_geocentric(
    x=-2059166.313 * un.m, y=-3621302.972 * un.m, z=4814304.113 * un.m
)
chime.info.name = "chime"

kko = ac.EarthLocation.from_geocentric(
    x = (-2111738.254-13.990) * un.m,
    y = (-3581458.222+4.666) * un.m, 
    z = (4821611.987-1.906) * un.m
)
kko.info.name = 'kko'
def generate_gaussian_signal(data,frame_start=100,frame_stop=1000):
    out_signal=np.zeros(data.ntime,dtype=data['baseband'].dtype)
    t_vals=np.arange(frame_start,frame_stop)
    center=(frame_stop+frame_start)//2
    width=(frame_start-frame_stop)//2
    out_signal[frame_start:frame_stop]+=(np.random.normal(0,1,len(t_vals))+1j*np.random.normal(0,1,len(t_vals)))*np.exp(-((t_vals-center)/width)**2)
    print(out_signal.shape)
    return out_signal
 

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
dec_cal2=58.24207034079757
cal_name2='J0137+3309'


#cal='J013658.5+475129'
#target='J0137+3309'




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

events=[350430021]




signal=None
event_id=350430021
#for event_id in events:

cleans=[False,True]

for clean in cleans:
    print(f"CLEAN: {clean}")
    
    event = get_event_data(event_id,events_database=kko_events_database,version='0.2test')
    src_name = event["source_name"][0]
    year = str(event["year"][0])
    month = int(event["month"][0])
    day = int(event["day"][0])
    ctime=float(event['ctime'][0])
    ratrue,dectrue=get_true_pulsar_pos(src_name,ctime=ctime)
    ra_src=ratrue+2 #2 degree diff
    dec_src=dectrue+10 #10 degrees off

    decs=[dec_src]
    ras=[ra_src]
    src_names=['fake_source']#src_name,cal_name,cal_name2]
    ras=np.array(ras)
    decs=np.array(decs)

    telescopes=['chime','kko']#,'chime']
    
    freqs_to_process=np.arange(0,1024,100)#None#np.arange(0,1024,1024)#$505)
    valid_freqs=[]
    for freq in freqs_to_process:
        files= glob(f'/arc/projects/chime_frb/vlbi/chime_rawdata/astro_{event_id}/baseband_{event_id}_{freq}*')
        if len(files)>0:
            valid_freqs.append(freq)
    print(valid_freqs)
    freqs_to_process=valid_freqs
    s_t_n=5
    for telescope in telescopes:
                
        for source in range(len(ras)):

            raw_data_dir=f'/arc/projects/chime_frb/temp/data/{telescope}/baseband/raw/*/*/*/astro_{event_id}/*.h5'
            print(raw_data_dir)
            datafiles_all=glob(raw_data_dir)
            if len(datafiles_all)==0 and telescope=='chime':
                raw_data_dir=f'/arc/projects/chime_frb/vlbi/chime_rawdata/astro_{event_id}/*.h5'
                datafiles_all=glob(raw_data_dir)

            datafiles_all.sort(key=freq_id_from_filename)
            if freqs_to_process is not None:
                new_files=[]
                for filename in datafiles_all:
                    if freq_id_from_filename(filename) in freqs_to_process:
                        new_files.append(filename)
                print(new_files)
                datafiles_all=new_files
                
            first_data = bbdata.BBData.from_acq_h5(datafiles_all[0])
            ctime = first_data["time0"]["ctime"][0]
            ratrue,dectrue=get_true_pulsar_pos(src_name,ctime=ctime)
            ra_src=ratrue
            dec_src=dectrue


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
            inputs = tools.get_correlator_inputs(date, correlator=backend.correlator)
            gains = cal.read_gains(cal_h5)
        
            from simulate_signal import get_input_delays
            
            # order feed positions (ordered to bbdata)
            converted_data_inputs = first_data.input.astype(
                    [("chan_id", "<u2"), ("correlator_input", "U32")]
                )
            reordered_inputs = tools.reorder_correlator_inputs(
                        converted_data_inputs, inputs
                    )
            if telescope=='chime':
                station=chime
            if telescope=='kko':
                station=kko
            
            feed_positions=convert_inputs_to_stations(reordered_inputs,telescope=station) #inputs should be ordered to correspond to baseband data
            mean_delays=get_input_delays(reference_station=chime,feed_positions=feed_positions,ctime=ctime,ra=ra_src,dec=dec_src)
            get_input_delays(feeds,station,ctime,ra,dec,static_delays)
            
            chunk_size=min(2,len(datafiles_all))
            frame_start=0
            frame_stop=10000
            print(f"CHUNKS: {len(datafiles_all)//chunk_size}")
            for i in range(len(datafiles_all)//chunk_size):
                print(f"{i*chunk_size}-{i*chunk_size+chunk_size} of {len(datafiles_all)} for telescope {telescope}!")
                files_to_process=datafiles_all[i*chunk_size:i*chunk_size+chunk_size]

                try:
                    data = bbdata.BBData.from_acq_h5(files_to_process)
                    if signal is None:
                        print("NO SIGNAL")
                        signal=generate_gaussian_signal(data,frame_start=frame_start,frame_stop=frame_stop)
                    else:
                        print("YES SIGNAL")
                    cal.apply_calibration(data, gains, inputs=inputs)
                    if True:
                        clean_persistent_rfi(
                            data=data, ra=np.array([ras[source]]), dec=np.array([decs[source]]), 
                            signal=signal,s_t_n=s_t_n,
                            ps_gain_file=cal_h5,inputs=inputs,
                            reference_feed=reference_feed,static_delays=static_delays,telescope=telescope,
                            obs=backend.obs,clean=clean,mean_delays=mean_delays,
                        frame_start=frame_start,frame_stop=frame_stop)


                    datas.append(data)

                except:
                    for j in range(len(files_to_process)):#datafiles_all[i*chunk_size:i*chunk_size+chunk_size])):
                        file=files_to_process[j]
                        if signal is None:
                            signal=generate_gaussian_signal(data,frame_start=frame_start,frame_stop=frame_stop)
                        inputs = tools.get_correlator_inputs(date, correlator=backend.correlator)
                        gains = cal.read_gains(cal_h5)
                        cal.apply_calibration(data, gains, inputs=inputs)
                        clean_persistent_rfi(
                            data=data, ra=np.array([ras[source]]), dec=np.array([decs[source]]), signal=signal,ps_gain_file=cal_h5,inputs=inputs,
                            reference_feed=reference_feed,static_delays=static_delays,telescope=telescope,
                            obs=backend.obs,mean_delays=mean_delays,
                frame_start=frame_start,frame_stop=frame_stop,clean=clean)

                        del data["baseband"]
                        datas.append(data)


            beamformed_rfi_cleaned = bbdata.concatenate(datas)
            print(out_file)
            beamformed_rfi_cleaned.save(out_file)
            beamformed_rfi_cleaned.close()
            #datatrail_pull_or_clear(event_id, telescope=telescope, cmd_str='CLEARED')
       