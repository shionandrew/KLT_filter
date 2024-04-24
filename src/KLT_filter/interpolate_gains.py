import baseband_analysis
from ch_util.cal_utils import interpolate_gain
import h5py
import chime_frb_api
from matplotlib import pyplot as plt
from baseband_analysis.core import BBData
import numpy as np
from glob import glob
from coda.core import VLBIVis
import pandas as pd
import astropy.units as u
import os
from astropy.coordinates import SkyCoord
from outriggers_vlbi_pipeline.diagnostic_plots import get_subframe_snr
from outriggers_vlbi_pipeline.query_database import update_event_status,get_event_data, get_full_filepath, find_files,fetch_data_from_sheet,check_correlation_completion,get_target_vis_files,get_cal_vis_files
from outriggers_vlbi_pipeline.vlbi_pipeline_config import chime, kko,calibrator_database, credentials_file, calibrator_database,kko_events_database,frb_events_database
from coda.analysis import cal
import parser
from typing import List
import argparse
from outriggers_vlbi_pipeline.cross_correlate_data import flag_rfi
from pyfx.core_vis import extract_frame_delay, extract_subframe_delay
from outriggers_vlbi_pipeline.diagnostic_plots import plot_localization,plot_visibility_diagnostics
from outriggers_vlbi_pipeline.query_database import get_baseband_localization_info
from multiprocessing import Pool
from outriggers_vlbi_pipeline.vlbi_pipeline_config import chime,kko,chime_obs,kko_obs
import time
import numpy as np
from glob import glob
from outriggers_vlbi_pipeline.query_database import get_baseband_localization_info
from outriggers_vlbi_pipeline.geometry import get_diagonal_grid_2
from glob import glob
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import logging
import pandas as pd
import json
from beam_model.utils import get_position_from_equatorial
from astropy.coordinates import SkyCoord
import numpy as np
import os
import math
from astropy import units as u
from typing import Tuple, Optional, Union, List
from astropy.time import Time
from coda.core import VLBIVis
from coda.analysis.flagging import undo_channel_mask
from outriggers_vlbi_pipeline.diagnostic_plots import plot_visibility_diagnostics,waterfall_pulsar
from coda.analysis.delay import get_subframe_snr, _get_subframe_snr
import matplotlib.pyplot as plt
from outriggers_vlbi_pipeline.vlbi_pipeline_config import calibrator_catalogue, calibrator_database, current_calibrators,current_version,known_pulsars
from outriggers_vlbi_pipeline.query_database import get_full_filepath, find_files
from outriggers_vlbi_pipeline.query_database import check_correlation_completion, update_event_status, get_event_data,fetch_data_from_sheet
from outriggers_vlbi_pipeline.vlbi_pipeline_config import kko_events_database,chime,kko,gbo
from outriggers_vlbi_pipeline.calibration import get_calibrator_visibilities, make_calibrated_visibilities
from outriggers_vlbi_pipeline.query_database import find_files, get_baseband_localization_info
from outriggers_vlbi_pipeline.geometry import get_diagonal_grid, angular_distance
from outriggers_vlbi_pipeline.localization.localization_model import localization_model
from outriggers_vlbi_pipeline.diagnostic_plots import plot_localization_search
from outriggers_vlbi_pipeline.geometry import get_1D_error
import logging
from outriggers_vlbi_pipeline.known_calibrators import get_true_pulsar_pos
import copy
from outriggers_vlbi_pipeline.vlbi_pipeline_config import chime, kko,calibrator_database, credentials_file, calibrator_database,kko_events_database,frb_events_database
import pickle
#from outriggers_vlbi_pipeline.scripts.run_pycalc_localization import localization_model
import re
from outriggers_vlbi_pipeline.localization.localization_model import localization_model
from outriggers_vlbi_pipeline.calibration import get_calibrator_visibilities,remove_rfi_all_sigmas
from outriggers_vlbi_pipeline.localization.localization_model import localization_model
from outriggers_vlbi_pipeline.query_database import get_baseband_localization_info
from astropy.time import Time
from outriggers_vlbi_pipeline.vlbi_pipeline_config import chime,kko,hco
from glob import glob
from baseband_analysis.core.bbdata import BBData


from scipy.optimize import curve_fit
import logging
import caput
import numpy as np
from typing import Optional
from astropy.coordinates import SkyCoord
import scipy
"""contains functions for localization/calculating localization precision"""
import logging
import coda
import pandas as pd
import numpy as np
from numpy import pi
from typing import Optional, Union, Tuple
import astropy
import astropy.units as un
import astropy.units as u
import astropy.coordinates as ac
from astropy.time import Time
from coda.core.math import complex_nanstd
from coda.analysis import cal
from typing import List
from glob import glob
from coda.core import VLBIVis
import coda.analysis.delay as delay  # use ss_error_prop branch
import matplotlib.pyplot as plt
from matplotlib import ticker
import math
import caput
from astropy.coordinates import SkyCoord
import copy
from outriggers_vlbi_pipeline.vlbi_pipeline_config  import (
    chime,
    kko,
    current_version,
    chime_obs,
    kko_obs
)
from outriggers_vlbi_pipeline.geometry import get_diagonal_grid, angular_distance, Gauss
from outriggers_vlbi_pipeline.query_database import find_files, get_full_filepath
from outriggers_vlbi_pipeline.known_calibrators import add_cal_status_to_catalogue
from outriggers_vlbi_pipeline.diagnostic_plots import (
    plot_visibility_diagnostics,
    get_subframe_snr,
)
from pycalc11 import Calc
from scipy.optimize import curve_fit
from coda.analysis.error_prop import get_vis_std_lag
from coda.analysis.cal import _eval_cpx_univariate_spline
from scipy.interpolate import UnivariateSpline
from coda.analysis import delay
from coda.analysis.delay import extract_subframe_delay, extract_frame_delay
from coda.core.math import _scrunch
from coda.core import VLBIVis, baseline
from coda.analysis import cal
from coda.analysis.cal import get_phase_template
from coda.analysis.cal import apply_phase_cal as apply_phase_cal
from numpy import deg2rad 
from numpy import rad2deg 
import pickle



def _eval_cpx_univariate_spline(x, y, s=.5):#0):#.5):
    """Sorts, calculates complex spline, and returns a function to evaluate.

    Parameters
    ----------
    x : np.array (freqs)
        Abscissas
    y : np.array (complex #s)
        Ordinates
    s : float
        Smoothing factor of the spline
    """
    iisort = np.argsort(x)
    spl_real = UnivariateSpline(x[iisort], np.real(y[iisort]))#, w=0.5 * w[iisort])
    spl_imag = UnivariateSpline(x[iisort], np.imag(y[iisort]))#, w=0.5 * w[iisort])
    spl_real.set_smoothing_factor(s)
    spl_imag.set_smoothing_factor(s)
    return lambda t: spl_real(t) + 1j * spl_imag(t)

def get_dig_gain(digital_gain_file):
    from ch_util import ephemeris, tools, rfi, andata, data_index

    dg = andata.DigitalGainData.from_acq_h5(digital_gain_file)

    dig_gain = np.moveaxis(dg.gain[:], 0, -1) #switch 0 axis and last axis

    #dig_gain.shape
    inputs_to_keep=[]
    for i in range((dig_gain.shape[1])):
        if dig_gain[10][i][0]!=0:
            inputs_to_keep.append(i)

    dig_gain_new=np.zeros((len(dig_gain),len(inputs_to_keep)),dtype=dig_gain.dtype)
    for i in range(len(inputs_to_keep)):
        input_index=inputs_to_keep[i]
        dig_gain_new[:,i]=dig_gain[:,input_index,0]

    return dig_gain_new
    
def interpolate_gains(
    gain_file,
    dig_gain_new=None):

    gain = baseband_analysis.core.calibration.read_gains(gain_file) #nfreq,ninputs

    f = h5py.File(gain_file, 'r')
    list(f.keys()) 
    weight=copy.deepcopy(f['weight'][:])
    freqs=copy.deepcopy(f['index_map']['freq'][:])['centre']

    f.close()

    try:
        interpolated_gains, interp_weight=interpolate_gain(freq=freqs,gain=gain,weight=weight)
    except:
        try:
            interpolated_gains, interp_weight=interpolate_gain(freq=freqs,gain=gain,weight=weight,length_scale=60.0)
        except:
            interpolated_gains, interp_weight=interpolate_gain(freq=freqs,gain=gain,weight=weight,length_scale=100.0)
    return interpolated_gains,gain

def interpolate_chime_gains(
        gain_file):
    gains = baseband_analysis.core.calibration.read_gains(gain_file) #nfreq,ninputs
    freq_ids=np.linspace(800,400,1024)
    existing_vals=np.where(np.max(np.abs(gains),axis=-1)!=0.0)
    zeroed_vals=np.where(np.max(np.abs(gains),axis=-1)==0.0)
    freqs=np.linspace(800,400,1024)    
    interpolated_gains=copy.deepcopy(gains)#np.zeros(gains.shape,dtype=gains.dtype)

    for antenna in range(gains.shape[-1]):    
        if antenna%10==0:
            print(f'interpolated gains for {antenna}/{gains.shape[-1]} antennas')
        if True:#dig_gain_new is None:
            spl=_eval_cpx_univariate_spline(x=freqs[existing_vals],y=gains[existing_vals][:,antenna])
            min_index=np.where(freqs==max(freqs[existing_vals]))[0][0]
            max_index=np.where(freqs==min(freqs[existing_vals]))[0][0] #avoid extrapolating fit
            interpolated_gains[:,antenna]=spl(freqs)
        else:
            spl=_eval_cpx_univariate_spline(x=freqs[existing_vals],y=gains[existing_vals][:,antenna]*dig_gain_new[existing_vals][:,antenna])
            min_index=np.where(freqs==max(freqs[existing_vals]))[0][0]
            max_index=np.where(freqs==min(freqs[existing_vals]))[0][0] #avoid extrapolating fit
            interpolated_gains[:,antenna]=spl(freqs)/dig_gain_new[:,antenna]

    return interpolated_gains,gains





import h5py
def make_interpolated_gain_file(cal_h5,telescope):
    gain_tag=re.split('.h5',cal_h5)[0]
    gain_tag=re.split('hdf5_files/',gain_tag)[1]
    out_file=f'/arc/home/shiona/{telescope}_interpolated_gains/{gain_tag}_interpolated.h5'
    print(out_file)
    if telescope=='chime':
        interpolated_gains,gains=interpolate_chime_gains(cal_h5)
    else:
        try:
            interpolated_gains,gains=interpolate_gains(cal_h5)
        except:
            out_file=f'/arc/home/shiona/{telescope}_interpolated_gains/{gain_tag}_spline_interpolated.h5'
            interpolated_gains,gains=interpolate_chime_gains(cal_h5)
    cal_5_copy=h5py.File(out_file, "w")#["gain"][:]
    cal_5_copy['gain']=interpolated_gains
    cal_5_copy.close()
    return out_file
