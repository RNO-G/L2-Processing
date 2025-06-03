import argparse
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.RNO_G.dataProviderRNOG
import snr
import rpr
import reco
import csw
import hilbert
import impulsivity
import utils
import NuRadioReco.detector.RNO_G.rnog_detector
from NuRadioReco.framework.parameters import eventParametersRNOG as ep
import logging
from NuRadioReco.examples.RNOG.processing import process_event
import NuRadioReco.modules.io.eventWriter
import matplotlib.pyplot as plt
from NuRadioReco.utilities import fft as fft_reco
import numpy as np 
import os 
import pandas as pd 
import csv 
import time
import datetime
from NuRadioReco.detector.detector import Detector
from NuRadioReco.modules.io import eventReader
import glob 
import os 
from NuRadioReco.detector.RNO_G import rnog_detector
import datetime as dt
import NuRadioReco.modules.channelAddCableDelay
channelCableDelayAdder = NuRadioReco.modules.channelAddCableDelay.channelAddCableDelay()
channelCableDelayAdder.begin()
from NuRadioReco.detector.RNO_G import rnog_detector
import logging 
import datetime as dt
import NuRadioReco.modules.channelSignalReconstructor
import scipy 

channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor(log_level=logging.WARNING)
channelSignalReconstructor.begin()

from NuRadioReco.utilities.framework_utilities import get_averaged_channel_parameter
from NuRadioReco.framework.parameters import (
    eventParameters as evp, channelParameters as chp, showerParameters as shp,
    particleParameters as pap, generatorAttributes as gta)

detectorpath = "/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/calib_nuradio.json"

csw_info = {
  "PA" : [0,1,2,3],
  "PS" : [0,1,2,3,5,6,7],
  "ALL" : [0,1,2,3,5,6,7,9,10,22,23],
  "Special" : [0,1,2,3,5,6,7,22,23]
}

do_envelope = True
res = 500
solution = "direct_combined"

rpr = rpr.RPR()
csw = csw.CSW()
reco = reco.Reco()
snr = snr.SNR()
hilbert = hilbert.Hilbert()
impulsivity = impulsivity.Impulsivity()

parser = argparse.ArgumentParser(description='L2')
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--stat', type=int, required=True)
parser.add_argument('--vert', type=str, required=False)
parser.add_argument('--nargs', nargs='+', type=int)
parser.add_argument('--pair', nargs='+', type=int)
args = parser.parse_args()
filename = args.file
station_id = args.stat
events = args.nargs
pair = args.pair
vert = args.vert
run_no = filename.split("/")[-1]
count_res = filename.split("/")[-1].split("_")[-1].split(".")[0]

"""
radii = {}
row_count = 0
with open(vert, mode = "r") as file:
    csvFile = csv.reader(file)
    for row in csvFile:
        if (row_count > 0):
            radii[row_count - 1] = round(float(row[3]))
        row_count += 1
"""
#outfile12 = f'/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/{station_id}_map_dc_large_0527.pkl'
outfile13 = f'/data/i3store/users/avijai/ttimes/'



ttcs2 = utils.load_ttcs(outfile13, csw_info["ALL"])



det = NuRadioReco.detector.RNO_G.rnog_detector.Detector(always_query_entire_description=True, detector_file = detectorpath)
det.update(datetime.datetime(2024, 3, 1))

dataProviderRNOG = NuRadioReco.modules.RNO_G.dataProviderRNOG.dataProvideRNOG()
dataProviderRNOG.begin(files = filename, det = det)


eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
eventWriter.begin(filename=f'/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/L2_files_test_0603/L2_{station_id}_{count_res}.nur')


def plot_trace(event, station, x):
    #x = event.get_id()
    power_str = [7,6,5,4,8,3,2,1,0]
    helper_str_1 = [11,10,9]
    helper_str_2 = [21,22,23]
    surface = [12,13,14,15,16,17,18,19,20]

    fig, axs = plt.subplots(9, 3, figsize=(10, 20), sharex = True)
    all_power = {}
    for ch in station.iter_channels():
        trace = ch.get_trace()
        times = ch.get_times()

        power = np.sum(np.array(trace)**2) / len(np.array(trace))

        all_power[ch.get_id()] = [power] 

        n_samples = len(times)
        sampling_frequency = 1/(times[1] - times[0])
        spectrum = fft_reco.time2freq(trace, sampling_frequency)
        frequencies = np.fft.rfftfreq(n_samples, 1 / sampling_frequency)
        ch_id = ch.get_id()
        if (ch_id in power_str):
            axs[power_str.index(ch_id), 0].plot(times, trace)
            #axs[power_str.index(ch_id), 0].plot(frequencies, np.abs(spectrum))
            axs[power_str.index(ch_id), 0].set_title(f"Ch {ch_id}, V")
            #axs[power_str.index(ch_id), 0].set_xlim(0,1)
        if (ch_id in helper_str_1):
            axs[helper_str_1.index(ch_id) + 6, 1].plot(times, trace)
            #axs[helper_str_1.index(ch_id) + 6, 1].plot(frequencies, np.abs(spectrum))
            axs[helper_str_1.index(ch_id) + 6, 1].set_title(f"Ch {ch_id}, V")
            #axs[helper_str_1.index(ch_id) + 6, 1].set_xlim(0,1)
        if (ch_id in helper_str_2):
            axs[helper_str_2.index(ch_id) + 6, 2].plot(times, trace)
            #axs[helper_str_2.index(ch_id) + 6, 2].plot(frequencies, np.abs(spectrum))
            axs[helper_str_2.index(ch_id) + 6, 2].set_title(f"Ch {ch_id}, V")
            #axs[helper_str_2.index(ch_id) + 6, 2].set_xlim(0,1)
    
    for i in range(6):
        fig.delaxes(axs[i][1])
        fig.delaxes(axs[i][2])
    fig.text(0.5, 0.001, 'Time', ha='center')
    fig.text(0.01, 0.5, 'Trace', va='center', rotation='vertical')
    fig.suptitle(f"Event {x} Trace")
    fig.tight_layout()
    
    if (os.path.exists(f"/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/sim_nu_outliers/{run_no}") == False):
        os.makedirs(f"/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/sim_nu_outliers/{run_no}")
    
    fig.savefig(f"/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/sim_nu_outliers/{run_no}/evt_{x}_trace.png")
    
    plt.close()

    fig, axs = plt.subplots(3,3,figsize=(10,10), sharex = True)
    row = 0
    col = 0
    count = 0
    for ch in station.iter_channels():
        if (ch.get_id() in surface):
            trace = ch.get_trace()
            times = ch.get_times()
            n_samples = len(times)
            sampling_frequency = 1/(times[1] - times[0])
            spectrum = fft_reco.time2freq(trace, sampling_frequency)
            frequencies = np.fft.rfftfreq(n_samples, 1 / sampling_frequency)
            ch_id = ch.get_id()
            axs[row, col].plot(times, trace)
            #axs[row, col].plot(frequencies, np.abs(spectrum))
            #axs[row, col].set_xlim(0,1)
            axs[row, col].set_title(f"Ch {ch_id}, V")
            if (row == 2):
                row = 0
            else:
                row += 1 
            if (count == 2):
                count = 0
                col += 1
            else:
                count += 1

    fig.text(0.5, 0.001, 'Time', ha='center')
    fig.text(0.01, 0.5, 'Trace', va='center', rotation='vertical')
    fig.suptitle(f"Event {x} Trace", y=0.98)
    fig.tight_layout()
    
    
    if (os.path.exists(f"/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/sim_nu_outliers/{run_no}") == False):
        os.makedirs(f"/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/sim_nu_outliers/{run_no}")

    fig.savefig(f"/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/sim_nu_outliers/{run_no}/evt_{x}_trace_surf.png")
    
    plt.close()
    """
    if (os.path.exists(f"/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/cw_power/{run_no}") == False):
        os.makedirs(f"/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/cw_power/{run_no}")
    
    df = pd.DataFrame(all_power).transpose()
    df.to_csv(f"/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/cw_power/{run_no}/evt_{x}_2.csv")
    """

path = filename
reader = eventReader.eventReader()
reader.begin(path)
count = 0   
count_res = path.split("/")[-1].split("_")[-1].split(".")[0]

for idx, event in enumerate(dataProviderRNOG.run()):
#for event in reader.run():    
    if (event.get_id() in events):
        process_event(event, det, run_no)
        x = event.get_id()
        station = event.get_station(station_id)
        #channelCableDelayAdder.run(event, station, det, mode='subtract')
        #channelSignalReconstructor.run(event, station, det)
        #plot_trace(event, station, count)
        
        
        maxcorr_point, maxcorr, score, t_ab, surf_corr_ratio_rz, max_surf_corr_rz, surf_corr_ratio_2_rz, max_surf_corr_2_rz, maxcorr_point2, maxcorr2, max_r, max_z, max_z_2, max_r_2 = reco.run(event, station, detectorpath, station_id, csw_info["ALL"], do_envelope, res, ttcs2, run_no) 
        
            
        avg_snr = get_averaged_channel_parameter(event, chp.SNR, channels_to_include = csw_info["ALL"])
        

        avg_rpr = get_averaged_channel_parameter(event, chp.root_power_ratio, channels_to_include = csw_info["ALL"])


        
        
        csw_rpr = {}
        csw_snr = {}
        csw_hilbert_snr = {}
        csw_impulsivity = {}
        csw_peak = {}
        csw_power = {}

        for chan_combo in csw_info.keys():
            if (True == True):
                chans = csw_info[chan_combo]
                csw_times, csw_values = csw.run(event, station, detectorpath, station_id, chans, solution, ttcs2, maxcorr_point2, maxcorr2, score, t_ab)
                csw_snr[chan_combo], vpp, rms = snr.get_snr_single(csw_times, csw_values)
                csw_snr[chan_combo]  = snr.get_snr_single(csw_times, csw_values)
                csw_rpr[chan_combo] = rpr.get_single_rpr(csw_times, csw_values)
                csw_hilbert_snr[chan_combo] = hilbert.hilbert_snr(csw_values)
                csw_hilbert = np.abs(scipy.signal.hilbert(csw_values))
                csw_peak[chan_combo] = max(csw_hilbert)
                csw_power[chan_combo] = csw_times[np.argmax(np.array(csw_values)**2)]
                csw_impulsivity[chan_combo] = impulsivity.calculate_impulsivity_measures(csw_values)

        

        event.add_parameter_type(ep)
        event[ep.avg_snr] = avg_snr
        event[ep.avg_rpr] = avg_rpr
        event[ep.max_corr_coords] = [maxcorr_point, maxcorr_point2]
        event[ep.max_corr] = [maxcorr, maxcorr2]
        event[ep.csw_snr] = csw_snr
        event[ep.csw_rpr] = csw_rpr
        event[ep.csw_hilbert_snr] = csw_hilbert_snr
        event[ep.csw_impulsivity] = csw_impulsivity 
        event[ep.surf_corr_ratio] = [surf_corr_ratio_rz, surf_corr_ratio_2_rz]
        event[ep.max_surf_corr] = [max_surf_corr_rz, max_surf_corr_2_rz]
        event[ep.max_surf_corr_pos] = [(max_r, max_z), (max_r_2, max_z_2)]
        event[ep.csw_peak] = csw_peak
        event[ep.csw_power] = csw_power
        #event.set_id(count) 
        print(event.get_id(), "event") 
        
        eventWriter.run(event, det=None, mode={'Channels':False, "ElectricFields":False})
    count += 1 

dataProviderRNOG.end()
eventWriter.end()

