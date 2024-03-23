import argparse
import sys
import numpy as np
from glob import glob

from rfi_shion_version import clean_persistent_rfi
from rfi_shion_version_polarization import clean_persistent_rfi as clean_persistent_rfi_pol

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
from simulate_signal import inject_signal_into_baseband,generate_time_shifted_signal
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



def generate_gaussian_signal(
    data,
    frame_start=100,
    frame_stop=1000):
    out_signal=np.zeros(data.ntime,dtype=data['baseband'].dtype)
    t_vals=np.arange(frame_start,frame_stop)
    center=(frame_stop+frame_start)//2
    width=(frame_start-frame_stop)//2
    N=len(t_vals)
    signal=(np.random.normal(0,1,N)+1j*np.random.normal(0,1,N))*np.exp(-((t_vals-center)/width)**2)
    out_signal[frame_start:frame_stop]+=signal
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
    parser.add_argument("--start", help="integer start", type=int,default=0)
    parser.add_argument("--stop", help="integer stop", type=int,default=None)
    parser.add_argument("--event", help="event_id", type=int)
    parser.add_argument("--tec", help="tec", type=float,default=0.0)
    parser.add_argument("--clean", help="clean", type=str,default='clean')
    parser.add_argument("--pol", help="pol", type=str,default='True')


    cmdargs = parser.parse_args()
    telescope=cmdargs.tel
    start=cmdargs.start
    stop=cmdargs.stop
    event_id=cmdargs.event
    TEC=cmdargs.tec
    pol=cmdargs.pol
    clean=cmdargs.clean
    if clean=='clean' or clean=='True':
        cleans=[True]
        print("WILL CLEAN")
    elif clean=='both':
        cleans=[True,False]
        print("WILL CLEAN AND NOT CLEAN")
    else:
        cleans=[False]
        print("WILL NOT CLEAN")
    assert (telescope=='chime') or (telescope=='kko')

    import pandas
    if pol=="True" or pol=="true":
        POL=True 
    else:
        POL=False

    signal_file=f'/arc/home/shiona/KLT_filter/sim_data/{event_id}_signal_simple.csv'
    print(f'reading {signal_file}')
    try:
        df=pandas.read_csv(signal_file)
    except:
        from KLT_filter.injections.simulate_signal import generate_gaussian_signal
        files= glob(f'/arc/projects/chime_frb/vlbi/chime_rawdata/astro_{event_id}/baseband_{event_id}_*')
        if len(files)==0:
            files= glob(f'/arc/projects/chime_frb/temp/data/chime/baseband/raw/*/*/*/astro_{event_id}/baseband_{event_id}_*')
        data= bbdata.BBData.from_acq_h5(files[0])
        sig=generate_gaussian_signal(data,frame_start=0,frame_stop=25000)
        del data
        df=pandas.DataFrame(sig[0],columns=['signal'])
        df.to_csv(signal_file,index=False)
    
    signal_full=np.array(df['signal'][:].astype(complex))

    frame_start=min(np.where(np.abs(signal_full)>0.0)[0])
    frame_stop=max(np.where(np.abs(signal_full)>0.0)[0])
    print(frame_start)
    print(frame_stop)
    duration=frame_stop-frame_start

    signal_x=signal_full[frame_start:frame_stop]
    
    I=1
    p=0.9
    Q=0
    U=0.35
    assert p**2*I-U**2-Q**2>0, "please enter valid polarization parameters"
    V=np.sqrt(p**2*I-U**2-Q**2)
    Ex_Ex_th=(I+Q)/2
    Ey_Ey_th=(I-Q)/2
    Ex_Ey_th=(U+1j*V)/2
    Ey_Ex_th=(U-1j*V)/2
    
    signal_x*=np.sqrt(Ex_Ex_th)/np.sqrt(np.mean(signal_x*np.conj(signal_x)))
    C=Ey_Ex_th/Ex_Ex_th
    signal_y=C*signal_x

    Ey_Ey=np.mean(signal_y*np.conj(signal_y))
    Ex_Ey=np.mean(signal_x*np.conj(signal_y))
    Ey_Ex=np.mean(signal_y*np.conj(signal_x))
    Ex_Ex=np.mean(signal_x*np.conj(signal_x))
    
    print("Ex_Ex, theoretical, empirical")
    print(f'{Ex_Ex_th},{Ex_Ex}')
    print("Ey_Ey, theoretical, empirical")
    print(f'{Ey_Ey_th},{Ey_Ey}')
    print("Ex_Ey, theoretical, empirical")
    print(f'{Ex_Ey_th},{Ex_Ey}')
    print("Ey_Ex, theoretical, empirical")
    print(f'{Ey_Ex_th},{Ey_Ex}')

    signal_x_full=signal_full
    signal_y_full=signal_full
    signal_x_full[frame_start:frame_stop]=signal_x
    signal_y_full[frame_start:frame_stop]=signal_y
    signal=[signal_x_full,signal_y_full]

    event = get_event_data(event_id,events_database=kko_events_database,version='0.2test')
    src_name = event["source_name"][0]
    year = str(event["year"][0])
    month = int(event["month"][0])
    day = int(event["day"][0])
    ctime=float(event['ctime'][0])
    ratrue,dectrue=get_true_pulsar_pos(src_name,ctime=ctime)
    decs=np.linspace(10,80,5)#[dec]
    ras=[ratrue]*len(decs)
    ras=np.array(ras)
    decs=np.array(decs)
    power=100
    
    
    #s_t_ns=[45]*len(decs)
    s_t_ns=[100]*len(decs) #Jy
    
    if stop is None:
        stop=len(decs)
        
    for source in range(start,stop):
        ra=ras[source]
        dec=decs[source]
        s_t_n=s_t_ns[source]
        s_t_n2=s_t_n#s_t_n/np.sqrt(duration)
        
        telescopes=[telescope]
        freqs_to_process=np.arange(10,250,1)#1024,1)
        valid_freqs=[]
        for freq in freqs_to_process:
            files= glob(f'/arc/projects/chime_frb/vlbi/chime_rawdata/astro_{event_id}/baseband_{event_id}_{freq}*')
            if len(files)==0:
                files= glob(f'/arc/projects/chime_frb/temp/data/chime/baseband/raw/*/*/*/astro_{event_id}/baseband_{event_id}_{freq}*')
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
                clean_freq_ids=np.arange(700,800)
                if freqs_to_process is not None:
                    new_files=[]
                    for filename in datafiles_all:
                        freq_id=freq_id_from_filename(filename)
                        if freq_id in freqs_to_process:
                            new_files.append(filename)
                        if freq_id in clean_freq_ids:
                            clean_file=filename
                    datafiles_all=new_files
                    

                data_clean = bbdata.BBData.from_acq_h5(clean_file)
                ctime = data_clean["time0"]["ctime"][0]
                
                noise=np.nanstd((np.abs(np.nansum(data_clean['baseband'][0,:],axis=0))**2)[frame_start:frame_stop],axis=-1) #shape ntime, std over ntime after summing inputs 
                amplitude=np.sqrt(noise)*s_t_n/(np.sqrt(1000)*np.sqrt(len(new_files))) #approximately how many inputs there are in chime data, scale by n frequency channels 


                if telescope=='kko':
                    backend=kko_backend
                    reference_feed = tools.ArrayAntenna(
                            id=-1, slot=-1, powered=True, flag=True, pos=[0, 0, 0], delay=0
                    ) 
                    static_delays=False
                    try:
                        if day<10:
                            day_str='0'+str(day)
                        else:
                            day_str=str(day)
                        if month<10:
                            month_str='0'+str(month)
                        else:
                            month_str=str(month)

                        cal_h5=glob(f'/arc/home/shiona/kko_interpolated_gains_scaled/gain_{year}{month_str}{day_str}*.h5')[0]
                    except:
                        #raise AssertionError, f"no files found in /arc/home/shiona/kko_interpolated_gains_scaled/gain_{year}{month_str}{day_str}*.h5"
                        print(f"WARNING: no files found in /arc/home/shiona/kko_interpolated_gains_scaled/gain_{year}{month_str}{day_str}*.h5!!")
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

                if POL:
                    src_name+='_pol_'
                if clean:
                    out_file=f'{output_file_dir}{event_id}_{telescope}_singlebeam_{src_name}_clean.h5'
                else:
                    out_file=f'{output_file_dir}{event_id}_{telescope}_singlebeam_{src_name}.h5'
                print(out_file)


                datas=[]
                date = datetime.utcfromtimestamp(ctime)
                
                if telescope=='chime':
                    file='/arc/home/shiona/chime_inputs.pkl'
                    import pandas as pd
                    inputs = pd.read_pickle(file)
                else:
                    inputs = tools.get_correlator_inputs(date, correlator=backend.correlator)
                gains = cal.read_gains(cal_h5)
            
                from simulate_signal import get_input_delays
                
                # order feed positions (ordered to bbdata)
                converted_data_inputs = data_clean.input.astype(
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
                                
                # we ignore dispersive delay over the channel for now
                antenna_delays_1,baseline_delay=get_input_delays(feeds=reordered_inputs,station=station,ctime=ctime+10,ra=ra,dec=dec,static_delays=static_delays)
                antenna_delays,baseline_delay=get_input_delays(feeds=reordered_inputs,station=station,ctime=ctime,ra=ra,dec=dec,static_delays=static_delays)
                print(np.nanmax(np.abs(antenna_delays_1-antenna_delays)))
                total_delay=antenna_delays+baseline_delay
                chunk_size=min(8,len(datafiles_all))

                print(f"CHUNKS: {len(datafiles_all)//chunk_size}")
                for i in range(len(datafiles_all)//chunk_size):
                    print(f"{i*chunk_size}-{i*chunk_size+chunk_size} of {len(datafiles_all)} for telescope {telescope}!")
                    files_to_process=datafiles_all[i*chunk_size:i*chunk_size+chunk_size]
                    
                    data = bbdata.BBData.from_acq_h5(files_to_process)

                    # remove cable delays etc before injecting signal into baseband data 
                    cal.apply_calibration(data, gains, inputs=inputs)
                    
                    if telescope!='chime':
                        inject_signal_into_baseband(data,np.array(signal)*amplitude,delays=total_delay,s_t_n=s_t_n2,tec=TEC,corr_inputs=inputs,frame_start=frame_start,frame_stop=frame_stop)
                        #inject_signal_into_baseband_2(data,np.array(signal),delays=total_delay,s_t_n=s_t_n2,tec=TEC,inputs=inputs,frame_start=frame_start,frame_stop=frame_stop,power=power,
                        #digital_gain_file=digital_gain_file,gain_file=cal_h5)
                    else:
                        inject_signal_into_baseband(data,np.array(signal)*amplitude,delays=total_delay,s_t_n=s_t_n2,tec=0,corr_inputs=inputs,frame_start=frame_start,frame_stop=frame_stop)
                        #inject_signal_into_baseband_2(data,np.array(signal),delays=total_delay,s_t_n=s_t_n2,tec=0,inputs=inputs,frame_start=frame_start,frame_stop=frame_stop,power=power,
                        #digital_gain_file=digital_gain_file,gain_file=cal_h5)
                    if POL:
                        clean_persistent_rfi_pol(
                            data=data, ra=np.array([ra]), dec=np.array([dec]), ps_gain_file=cal_h5,inputs=inputs,
                            reference_feed=reference_feed,static_delays=static_delays,clean=clean,
                            obs=backend.obs,
                            Ex_Ex=Ex_Ex,
                            Ey_Ey=Ey_Ey,
                            Ex_Ey=Ex_Ey,
                            Ey_Ex=Ey_Ex,
                        frame_start=frame_start,frame_stop=frame_stop) # fringestop_delays=antenna_delays,

                    else:
                        clean_persistent_rfi(
                            data=data, ra=np.array([ra]), dec=np.array([dec]), ps_gain_file=cal_h5,inputs=inputs,
                            reference_feed=reference_feed,static_delays=static_delays,clean=clean,
                            obs=backend.obs,source_names=np.array(['placeholder_name']),
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



                beamformed_rfi_cleaned = bbdata.concatenate(datas)
                print(out_file)
                beamformed_rfi_cleaned.save(out_file)
                beamformed_rfi_cleaned.close()
                #datatrail_pull_or_clear(event_id, telescope=telescope, cmd_str='CLEARED')
        