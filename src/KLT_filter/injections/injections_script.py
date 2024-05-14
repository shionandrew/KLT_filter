import argparse
import sys
import numpy as np
from glob import glob

from KLT_filter.rfi_shion_version_old import clean_persistent_rfi
from KLT_filter.clean_beamform import beamform_clean_rfi as clean_persistent_rfi_pol
from beam_model.utils import get_equatorial_from_position,get_position_from_equatorial

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

home_dir='/home/shiona/'
processed_files_output_dir='/home/shiona/rfi_analysis_simulations/'

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

def undo_calibration(data, gains):
    """Apply calibration in `gains` derived from `calibrator` to a BBData object `data`.

    If inputs are supplied, will also compute phase center of telescope and append to
    data object.

    Input ordering convention: input_reorder takes an arbitrary ordering from the data.
    Gain ordering convention: assumed to be ordered by chan_id.

    Parameters
    ----------
    data: `BBData`
        Data to calibrate.
    gains: ndarray of complex128
        Complex input gains (n_freq, n_input)
        The array of complex gains as a function of frequency and correlator input
    inputs : list of ArrayAntenna objects
        If provided, will calculate the phase center of the telescope from the
        correlator inputs.
    calibrator : str
        A string denoting the N^2 calibrator used to solve for the telescope gains.

    Outputs
    -------
    Calibrated full-array baseband data.
    """
    freq_id_list = data.index_map["freq"]["id"]
    gains = gains[
        freq_id_list
    ]  
    input_reorder = data.input["chan_id"]
    gain_reordered = gains[:, input_reorder]
    data.baseband[:] /= np.conj(gain_reordered[:, :, np.newaxis])
    return

def perturb_gains(gains,gain_err):
    return gains+(np.random.normal(0,gain_err,gains.shape)+1j*np.random.normal(0,gain_err,gains.shape))*gains

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
    parser.add_argument("--event", help="event_id", type=int)
    parser.add_argument("--tec", help="tec (default 0)", type=float,default=0.0)
    parser.add_argument("--geo_delay", help="geo_delay (default 0)", type=float,default=0.0)
    parser.add_argument("--clean", help="clean (clean,both,all,false)", type=str,default='clean')
    parser.add_argument("--pol", help="pol, False, True", type=str,default='False')
    parser.add_argument("--dec", help="dec (default 10)", type=float,default=10.0)
    parser.add_argument("--flux", help="signal flux in Jy (default 1)", type=float,default=1.0)
    parser.add_argument("--freq_start", help="freq_start (default 10)", type=int,default=10)
    parser.add_argument("--freq_stop", help="freq_stop (default 1250)", type=int,default=250)
    parser.add_argument("--N_sources", help="number of sources if dec not specified", type=int,default=1)
    parser.add_argument("--randomize", help="number of sources if dec not specified", type=str,default='False')
    parser.add_argument("--polarization", help="linear or circular", type=str,default='linear')
    parser.add_argument("--p_val", help="polarization fraction", type=float,default=0.7)
    parser.add_argument("--gain_err", help="gain_err", type=float,default=0.0)
    parser.add_argument("--pol_err", help="pol_err", type=float,default=0.0)


    cmdargs = parser.parse_args()
    telescope=cmdargs.tel
    gain_err=cmdargs.gain_err
    event_id=cmdargs.event
    tec=cmdargs.tec
    geo_delay=cmdargs.geo_delay
    pol=cmdargs.pol
    clean=cmdargs.clean
    dec=cmdargs.dec
    flux=cmdargs.flux
    freq_start=cmdargs.freq_start
    freq_stop=cmdargs.freq_stop
    N_sources=cmdargs.N_sources
    randomize=cmdargs.randomize
    polarization=cmdargs.polarization
    p_val=cmdargs.p_val
    pol_err=cmdargs.pol_err
    
    import pandas
    if pol=="True" or pol=="true":
        POLS=[True] 
    else:
        POLS=[False]

    if randomize=="True":
        randomize=True
    else:
        randomize=False
    if clean=='clean' or clean=='True':
        cleans=[True]
        print("WILL CLEAN")
    elif clean=='both':
        cleans=[True,False]
        POLS=POLS*len(cleans)
        print("WILL CLEAN AND NOT CLEAN")
    elif clean=='all':
        cleans=[True,True,False]
        POLS=[True,False,False]
        print("WILL CLEAN POL, CLEAN, AND NOT CLEAN")
    else:
        cleans=[False]
        print("WILL NOT CLEAN")

    if telescope=='both':
        telescopes=['chime','kko']
    else:
        assert (telescope=='chime') or (telescope=='kko')
        telescopes=[telescope]




    print(f'{home_dir}KLT_filter/sim_data/{event_id}_signal_{p_val}*_{polarization}*_pol.csv')
    signal_file=glob(f'{home_dir}KLT_filter/sim_data/{event_id}_signal_{p_val}*_{polarization}*_pol.csv')[0]
    df=pandas.read_csv(signal_file)
    signal_x=np.array(df['signal_xx'][:].astype(complex))
    signal_y=np.array(df['signal_yy'][:].astype(complex))
    frame_start=df['frame_start'][0]
    frame_stop=df['frame_stop'][0]
    scale_factor=flux/(np.mean(np.abs(signal_x[frame_start:frame_stop]**2)))
    signal_x*=np.sqrt(scale_factor)
    signal_y*=np.sqrt(scale_factor)

    window=(frame_stop-frame_start)//4

    Ex_Ex=np.mean((signal_x*np.conj(signal_x))[frame_start+window:frame_stop-window])
    Ey_Ey=np.mean((signal_y*np.conj(signal_y))[frame_start+window:frame_stop-window])
    Ey_Ex=np.mean((signal_y*np.conj(signal_x))[frame_start+window:frame_stop-window])
    Ex_Ey=np.mean((signal_x*np.conj(signal_y))[frame_start+window:frame_stop-window])
    #signal=[signal_x,signal_y]
    signal=[signal_x,signal_y]
    
    def get_stokes(Ex,Ey):
        stokes_00=np.mean(Ex*np.conj(Ex))
        stokes_01=np.mean(Ex*np.conj(Ey))
        stokes_10=np.mean(Ey*np.conj(Ex))
        stokes_11=np.mean(Ey*np.conj(Ey))
        I=stokes_00+stokes_11
        Q=stokes_00-stokes_11
        U=stokes_01+stokes_10
        V=-1j*(stokes_01-stokes_10)
        #print(f"I: {I}, Q: {Q}, U: {U}, V: {V}")
        return I,Q,U,V
    I,Q,U,V=get_stokes(signal_x[frame_start:frame_stop],signal_y[frame_start:frame_stop])
    if pol_err>0:
        print("perturbing pol")
        V=max(1,np.abs(V)+pol_err*np.abs(V))
        Q=min(0,np.abs(Q)-pol_err*np.abs(Q))
        U=min(0,np.abs(U)-pol_err*np.abs(U))
        Ex_Ex=(I+Q)/2
        Ey_Ey=(I-Q)/2
        Ex_Ey=(U-1j*V)/2
        Ey_Ex=(U+1j*V)/2
    

    print(f"frame_start at {frame_start}")
    print(f"frame_stop at {frame_stop}")
    duration=frame_stop-frame_start
        

    event = get_event_data(event_id,events_database=kko_events_database,version='0.2test')
    src_name = event["source_name"][0]
    year = str(event["year"][0])
    month = int(event["month"][0])
    day = int(event["day"][0])
    ctime=float(event['ctime'][0])
    ratrue,dectrue=get_true_pulsar_pos(src_name,ctime=ctime)
    
    #print('true dec')
    #print(dectrue)
    #input("Press Enter to continue...")

    if N_sources>1 or randomize==True:
        fluxes=10*np.random.uniform(0.5,5,N_sources)
        ys=np.random.uniform(-60,60,N_sources)
        xs=np.random.uniform(-2,2,N_sources)
        ras,decs=get_equatorial_from_position(xs, ys, ctime)
        tecs=np.random.normal(0,1,N_sources)
        geo_delays=np.random.normal(0,1e-2,N_sources) #10ns sigma  
    else:
        decs=[dec]*N_sources
        ras=[ratrue]*N_sources
        fluxes=[flux]*N_sources
        tecs=[tec]*N_sources
        geo_delays=[geo_delay]*N_sources
    
    print(f"ras: {ras}")
    print(f"decs: {decs}")
    print(f"fluxes: {fluxes}")
    print(f"tecs: {tecs}")
    print(f"geo_delay: {geo_delays}")
    print(f"tels: {telescopes}")
    print(f"cleans: {cleans}")
    #input("Press Enter to continue...")


    for source in range(len(decs)):
        ra=ras[source]
        dec=decs[source]
        flux=fluxes[source]
        tec=tecs[source]
        geo_delay=geo_delays[source]

        x,y=get_position_from_equatorial(ra, dec, ctime)

        freqs_to_process=np.arange(freq_start,freq_stop,1)
        valid_freqs=[]
        for freq in freqs_to_process:
            files= glob(f'{home_dir}chime_rawdata/astro_{event_id}/baseband_{event_id}_{freq}*')
            if len(files)>0:
                valid_freqs.append(freq)
        print(valid_freqs)
        freqs_to_process=valid_freqs
        for telescope in telescopes:
            print(f"{telescope} out of {telescopes}")            
            src_name=f'fake_source_{ra}_{dec}_{flux}Jy'
            raw_data_dir=f'{home_dir}{telescope}_rawdata/astro_{event_id}/*.h5'
            print(raw_data_dir)
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
        
            if day<10:
                day_str='0'+str(day)
            else:
                day_str=str(day)
            if month<10:
                month_str='0'+str(month)
            else:
                month_str=str(month)
            print(f'/gain_{year}{month_str}{day_str}*.h5')
            if telescope=='kko':
                backend=kko_backend
                reference_feed = tools.ArrayAntenna(
                        id=-1, slot=-1, powered=True, flag=True, pos=[0, 0, 0], delay=0
                ) 
                static_delays=False

                cal_h5=glob(f'{home_dir}kko_interpolated_gains_scaled/gain_{year}{month_str}{day_str}*.h5')[0]

            elif telescope=='gbo':
                backend=gbo_backend
                reference_feed = tools.ArrayAntenna(
                        id=-1, slot=-1, powered=True, flag=True, pos=[0, 0, 0], delay=0
                ) 
                static_delays=False
                #cal_h5 = get_calfiles(day=day,month=month,year=year,telescope=telescope, ctime=ctime,gain_calibrator='casa')
                cal_h5=glob(f'{home_dir}gbo_interpolated_gains_scaled/gain_{year}{month_str}{day_str}*.h5')[0]

            else:
                backend=chime_backend
                reference_feed=tools.CHIMEAntenna(
                        id=-1, slot=-1, powered=True, flag=True, pos=[0, 0, 0]
                )
                static_delays = backend.static_delays

                #cal_h5 = get_calfiles(day=day,month=month,year=year,telescope=telescope, ctime=ctime)
                cal_h5=glob(f'{home_dir}chime_gains/gain_{year}{month_str}{day_str}*.h5')[0]
                print(cal_h5)

            import os
            output_file_dir=f'{processed_files_output_dir}{event_id}/{src_name}/{p_val}_{polarization}_polarization_tec_{tec}_geodelay_{geo_delay}_singlebeams/'
            print(output_file_dir)
            os.umask(0)
            os.makedirs(
                output_file_dir, exist_ok=True, mode=0o777
            )  # if output directory doesn't exist, create it


            date = datetime.utcfromtimestamp(ctime)
            
            import pandas as pd
            file=f'{home_dir}{telescope}_inputs.pkl'
            inputs = pd.read_pickle(file)

            gains = cal.read_gains(cal_h5)
            if gain_err>0:
                new_gains=perturb_gains(gains,gain_err)

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
            #antenna_delays_1,baseline_delay=get_input_delays(feeds=reordered_inputs,station=station,ctime=ctime+10,ra=ra,dec=dec,static_delays=static_delays)
            antenna_delays,baseline_delay=get_input_delays(feeds=reordered_inputs,station=station,ctime=ctime,ra=ra,dec=dec,static_delays=static_delays)

            #print(np.nanmax(np.abs(antenna_delays_1-antenna_delays)))
            total_delay=antenna_delays+baseline_delay
            chunk_size=min(1,len(datafiles_all))

            print(f"CHUNKS: {len(datafiles_all)//chunk_size}")
            for i in range(len(cleans)):
                clean=cleans[i]
                POL=POLS[i]
                print(f"POL: {POL}")
                print(f"CLEAN: {clean}")
                tag=''
                if clean:
                    tag='_clean'
                if POL:
                    tag+='_pol'
                    if pol_err>0:
                        tag+=f'_pol_err_{pol_err}'
                if gain_err>0:
                    tag+=f'_gain_err_{gain_err}'

                out_file=f'{output_file_dir}{event_id}_{telescope}_singlebeam_{src_name}{tag}_{freq_start}_{freq_stop}.h5'
                print(out_file)

                datas=[]
                for i in range(len(datafiles_all)//chunk_size):
                    print(f"{i*chunk_size}-{i*chunk_size+chunk_size} of {len(datafiles_all)} for telescope {telescope}!")
                    files_to_process=datafiles_all[i*chunk_size:i*chunk_size+chunk_size]
                    
                    data = bbdata.BBData.from_acq_h5(files_to_process)

                    # remove cable delays etc before injecting signal into baseband data 
                    cal.apply_calibration(data, gains, inputs=inputs)
                    print("signal")

                    if False:#null_background:
                        tag+=f'_null_background'

                        data['baseband'][:]*=0 #remove rfi, simulate noise
                        sigma=10
                        noise=np.random.normal(0,sigma,data['baseband'][:].shape)+1j*np.random.normal(0,sigma,data['baseband'][:].shape)
                        data['baseband'][:]+=noise

                        total_delay2,_=get_input_delays(feeds=reordered_inputs,station=station,ctime=ctime,ra=ra,dec=dec+1,static_delays=static_delays) #add very close in position

                        signal_file=glob(f'{home_dir}KLT_filter/sim_data/{event_id}_signal_{p_val}*_{polarization}*_pol.csv')[0]
                        df=pandas.read_csv(signal_file)
                        signal_x=np.array(df['signal_xx'][:].astype(complex))
                        signal_y=np.array(df['signal_yy'][:].astype(complex))
                        frame_start=df['frame_start'][0]
                        frame_stop=df['frame_stop'][0]
                        scale_factor=flux/(np.mean(np.abs(signal_x[frame_start:frame_stop]**2)))
                        signal_x*=np.sqrt(scale_factor)
                        signal_y*=np.sqrt(scale_factor)
                        signal=[signal_x,signal_y]

                        delayed_signals,noises=inject_signal_into_baseband(data,np.array(signal2),delays=total_delay2,geo_delay=0,tec=0,corr_inputs=inputs,frame_start=frame_start2,frame_stop=frame_stop2)
                    
                    _ = data.create_dataset(
                        "input_signal", shape=(data.baseband.shape), dtype=data.baseband.dtype
                    )
                    data['input_signal'][:]=0
                    if telescope!='chime':
                        delayed_signals,noises=inject_signal_into_baseband(data,np.array(signal),delays=total_delay,geo_delay=geo_delay,tec=tec,corr_inputs=inputs,frame_start=frame_start,frame_stop=frame_stop)
                    else:
                        delayed_signals,noises=inject_signal_into_baseband(data,np.array(signal),delays=total_delay,geo_delay=0,tec=0,corr_inputs=inputs,frame_start=frame_start,frame_stop=frame_stop)
                    
                    if gain_err>0:
                        print("perturbing gains!")
                        undo_calibration(data, gains)
                        cal.apply_calibration(data, new_gains)

                    if POL:
                        print("POL")
                        clean_persistent_rfi_pol(
                            data=data, ra=np.array([ra]), dec=np.array([dec]), inputs=inputs,
                            reference_feed=reference_feed,static_delays=static_delays,clean=clean,
                            fringestop_delays=antenna_delays,Ex_Ex=Ex_Ex,Ey_Ey=Ey_Ey,Ex_Ey=Ex_Ey,Ey_Ex=Ey_Ex,
                            obs=backend.obs,source_names=np.array(['placeholder_name']),POL=POL,gains=gains,
                        frame_start=frame_start,frame_stop=frame_stop)
                        
                    else:
                        clean_persistent_rfi(
                        data=data, ra=np.array([ra]), dec=np.array([dec]), ps_gain_file=cal_h5,inputs=inputs,
                        reference_feed=reference_feed,static_delays=static_delays,clean=clean,
                        fringestop_delays=antenna_delays,
                        obs=backend.obs,source_names=np.array(['placeholder_name']),#POL=POL,
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
                beamformed_rfi_cleaned.attrs['cal_h5']=cal_h5
                beamformed_rfi_cleaned.attrs['frame_start']=frame_start
                beamformed_rfi_cleaned.attrs['frame_stop']=frame_stop
                beamformed_rfi_cleaned.attrs['x_deg']=x
                beamformed_rfi_cleaned.attrs['y_deg']=y
                #if POL:
                beamformed_rfi_cleaned.attrs['polarization']=polarization
                beamformed_rfi_cleaned.attrs['p_frac']=p_val
                beamformed_rfi_cleaned.save(out_file)
                beamformed_rfi_cleaned.close()
                #datatrail_pull_or_clear(event_id, telescope=telescope, cmd_str='CLEARED')
        