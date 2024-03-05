import argparse
import sys
import numpy as np
from glob import glob

from rfi_shion_version_test import clean_persistent_rfi

import baseband_analysis.core.calibration as cal
import copy
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
from simulate_signal import convert_inputs_to_stations,inject_signal_into_baseband,generate_time_shifted_signal
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
        
if __name__=='__main__': 
    parser = argparse.ArgumentParser("rfi Executable")
    parser.add_argument("--tel", help="telescope (chime or kko)", type=str)
    parser.add_argument("--start", help="integer start", type=int)
    parser.add_argument("--stop", help="integer stop", type=int)
    parser.add_argument("--event", help="event_id", type=int)
    cmdargs = parser.parse_args()
    telescope=cmdargs.tel
    start=cmdargs.start
    stop=cmdargs.stop
    event_id=cmdargs.event
    assert (telescope=='chime') or (telescope=='kko')

    signal=None
    #event_id=350430021 #308984712, 313966747, 314210443, 339559697
    import pandas
    signal_file=f'/arc/home/shiona/calibrator_survey/rfi_analysis/{event_id}_signal.csv'
    print(f'writing to {signal_file}')
    try:
        df=pandas.read_csv(signal_file)
        signal=np.array(df['signal'][:].astype(complex))
    except:
        signal=None
    event = get_event_data(event_id,events_database=kko_events_database,version='0.2test')
    src_name = event["source_name"][0]
    year = str(event["year"][0])
    month = int(event["month"][0])
    day = int(event["day"][0])
    ctime=float(event['ctime'][0])
    ratrue,dectrue=get_true_pulsar_pos(src_name,ctime=ctime)
    decs=np.linspace(10,80,50)#[dec]
    ras=[ratrue]*len(decs)
    ras=np.array(ras)
    decs=np.array(decs)


    cleans=[False,True]
    
    s_t_ns=np.linspace(.2,1,len(decs))
    
    frame_start=0
    frame_stop=20000
    duration=frame_stop-frame_start
    for source in range(start,stop):
        ra=ras[source]
        dec=decs[source]
        s_t_n=s_t_ns[source]
        s_t_n2=s_t_n/np.sqrt(duration)
        
        telescopes=[telescope]#'kko']#,'kko']#,'chime']#,'kko']#,'chime']
        freqs_to_process=np.arange(0,1024)#None#np.arange(0,1024,1024)#$505)
        valid_freqs=[]
        for freq in freqs_to_process:
            files= glob(f'/arc/projects/chime_frb/vlbi/chime_rawdata/astro_{event_id}/baseband_{event_id}_{freq}*')
            if len(files)>0:
                valid_freqs.append(freq)
        print(valid_freqs)
        freqs_to_process=valid_freqs
        for clean in cleans:
            print(f"CLEAN: {clean}")
            for telescope in telescopes:
                src_name=f'fake_source_{ra}_{dec}_{s_t_n}'
                raw_data_dir=f'/arc/projects/chime_frb/temp/data/{telescope}/baseband/raw/*/*/*/astro_{event_id}/*.h5'
                print(raw_data_dir)
                datafiles_all=glob(raw_data_dir)
                if len(datafiles_all)==0:
                    raw_data_dir=f'/arc/projects/chime_frb/vlbi/{telescope}_rawdata/astro_{event_id}/*.h5'
                    datafiles_all=glob(raw_data_dir)
                    assert len(datafiles_all)>0

                datafiles_all.sort(key=freq_id_from_filename)
                if freqs_to_process is not None:
                    new_files=[]
                    for filename in datafiles_all:
                        if freq_id_from_filename(filename) in freqs_to_process:
                            new_files.append(filename)
                    datafiles_all=new_files
                    
                first_data = bbdata.BBData.from_acq_h5(datafiles_all[0])
                ctime = first_data["time0"]["ctime"][0]


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
                    out_file=f'{output_file_dir}{event_id}_{telescope}_singlebeam_{src_name}_clean.h5'
                else:
                    out_file=f'{output_file_dir}{event_id}_{telescope}_singlebeam_{src_name}.h5'
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

                reordered_inputs.append(reference_feed)

                if telescope=='chime':
                    station=chime
                    static_delays=True
                if telescope=='kko':
                    station=kko
                    static_delays=False
                
                #feed_positions=convert_inputs_to_stations(reordered_inputs,telescope=station) #inputs should be ordered to correspond to baseband data
                
                # we ignore dispersive delay over the channel for now
                mean_delays,baseline_delay=get_input_delays(feeds=reordered_inputs,station=station,ctime=ctime,ra=ra,dec=dec,static_delays=static_delays)
                mean_delays+=baseline_delay
                chunk_size=min(1,len(datafiles_all))

                print(f"CHUNKS: {len(datafiles_all)//chunk_size}")
                for i in range(len(datafiles_all)//chunk_size):
                    print(f"{i*chunk_size}-{i*chunk_size+chunk_size} of {len(datafiles_all)} for telescope {telescope}!")
                    files_to_process=datafiles_all[i*chunk_size:i*chunk_size+chunk_size]
                    
                    if True:
                        data = bbdata.BBData.from_acq_h5(files_to_process)
                        if signal is None:
                            print("NO SIGNAL")
                            signal=generate_gaussian_signal(data,frame_start=frame_start,frame_stop=frame_stop)
                            print(f'writing to {signal_file}')
                            df=pandas.DataFrame(signal,columns=['signal'])
                            df.to_csv(signal_file,index=False)

                        else:
                            print("YES SIGNAL")
                        cal.apply_calibration(data, gains, inputs=inputs)
                        
                        inject_signal_into_baseband(data,signal,delays=mean_delays,s_t_n=s_t_n2)

                        clean_persistent_rfi(
                            data=data, ra=np.array([ra]), dec=np.array([dec]), ps_gain_file=cal_h5,inputs=inputs,
                            reference_feed=reference_feed,static_delays=static_delays,clean=clean,
                            obs=backend.obs,
                        frame_start=frame_start,frame_stop=frame_stop)


                        if telescope!='chime':
                            #fringestop data to chime
                            print("fringe stopping")
                            fringedstopped_tiedbeam=generate_time_shifted_signal(
                                copy.deepcopy(data['tiedbeam_baseband'][:,:,:]),
                                -baseline_delay, #microseconds
                                data.freq, #Mhz, either float or array of floats
                                beamformed=True,
                                )
                            data['tiedbeam_baseband'][:,:,:]=fringedstopped_tiedbeam

                        datas.append(data)

                    '''except:
                        for j in range(len(files_to_process)):#datafiles_all[i*chunk_size:i*chunk_size+chunk_size])):
                            file=files_to_process[j]
                            if signal is None:
                                signal=generate_gaussian_signal(data,frame_start=frame_start,frame_stop=frame_stop)
                            inputs = tools.get_correlator_inputs(date, correlator=backend.correlator)
                            gains = cal.read_gains(cal_h5)
                            cal.apply_calibration(data, gains, inputs=inputs)
                            clean_persistent_rfi(
                                data=data, ra=np.array([ra]), dec=np.array([dec]), ps_gain_file=cal_h5,inputs=inputs,
                                reference_feed=reference_feed,static_delays=static_delays,clean=clean,
                                obs=backend.obs,
                            frame_start=frame_start,frame_stop=frame_stop)
                            datas.append(data)'''
                    

                beamformed_rfi_cleaned = bbdata.concatenate(datas)
                print(out_file)
                beamformed_rfi_cleaned.save(out_file)
                beamformed_rfi_cleaned.close()
                #datatrail_pull_or_clear(event_id, telescope=telescope, cmd_str='CLEARED')
        