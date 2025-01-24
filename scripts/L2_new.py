import argparse
import preprocessor 
import NuRadioReco.modules.channelSignalReconstructor
import snr
import rpr
import reco
import csw
import hilbert
import impulsivity
import utils
import NuRadioReco.detector.RNO_G.rnog_detector
from NuRadioReco.framework.parameters import eventParameters as ep
import logging

detectorpath = "/data/i3store/users/avijai/RNO_season_2023.json"
csw_info = {
  "PA" : [0,1,2,3],
  "PS" : [0,1,2,3,5,6,7],
  "ALL" : [0,1,2,3,5,6,7,9,10,22,23]
}

do_envelope = True
res = 100
solution = "direct_ice"

rpr = rpr.RPR()
csw = csw.CSW()
reco = reco.Reco()
snr = snr.SNR()
hilbert = hilbert.Hilbert()
impulsivity = impulsivity.Impulsivity()

parser = argparse.ArgumentParser(description='L2')
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--stat', type=int, required=True)
args = parser.parse_args()
filename = args.file
station_id = args.stat


mappath = reco.build_travel_time_maps(detectorpath, station_id, csw_info["ALL"])
ttcs = utils.load_ttcs(mappath, csw_info["ALL"])

det = NuRadioReco.detector.RNO_G.rnog_detector.Detector()

channelSignalReconstructor = NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
channelSignalReconstructor.begin()

preprocessor = preprocessor.Preprocessor()
preprocessor.begin()


readRNOGDataMattak = NuRadioReco.modules.io.RNO_G.readRNOGDataMattak.readRNOGData(log_level=logging.INFO)
readRNOGDataMattak.begin(filename, read_calibrated_data = False, convert_to_voltage = True, overwrite_sampling_rate = 3.2)

for idx, event in enumerate(readRNOGDataMattak.run()):
    station = event.get_station(station_id)
    preprocessor.run(event, station, det)

    det.update(station.get_station_time())

    maxcorr_point, maxcorr, score, t_ab, surf_corr_ratio, max_surf_corr = reco.run(event, station, detectorpath, station_id, csw_info["ALL"], do_envelope, res, mappath, ttcs) 
    channelSignalReconstructor.run(event, station, det)
    avg_snr = event.avg_SNR()
    avg_rpr = event.avg_RPR()
    
    csw_rpr = {}
    csw_snr = {}
    csw_hilbert_snr = {}
    csw_impulsivity = {}

    for chan_combo in csw_info.keys():
        chans = csw_info[chan_combo]
        csw_times, csw_values = csw.run(event, station, detectorpath, station_id, chans, solution, ttcs, maxcorr_point, maxcorr, score, t_ab)
        csw_snr[chan_combo] = snr.get_snr_single(csw_times, csw_values)
        csw_rpr[chan_combo] = rpr.get_single_rpr(csw_times, csw_values)
        csw_hilbert_snr[chan_combo] = hilbert.hilbert_snr(csw_values)
        csw_impulsivity[chan_combo] = impulsivity.calculate_impulsivity_measures(csw_values)

    
    event[ep.max_corr_coords] = maxcorr_point 
    event[ep.max_corr] = maxcorr
    event[ep.csw_snr] = csw_snr
    event[ep.csw_rpr] = csw_rpr
    event[ep.csw_hilbert_snr] = csw_hilbert_snr
    event[ep.csw_impulsivity] = csw_impulsivity 
    event[ep.surf_corr_ratio] = surf_corr_ratio
    event[ep.max_surf_corr] = max_surf_corr

