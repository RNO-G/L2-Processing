import defs
from NuRadioReco.detector.RNO_G import rnog_detector
from detector import get_channel_positions,calculate_avg_antenna_xyz,get_cable_delays, get_device_position
from propagation import TravelTimeCalculator
import pickle
import datetime
import utils
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RectBivariateSpline
from numpy import gradient, sqrt
from scipy.linalg import svd
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
import zarr 
import logging 
from NuRadioReco.detector import detector

class Reco:

    def __init__(self):
        self.num_pts_z = 6000
        self.num_pts_r = 6000
        self.ior_model = defs.ior_exp1 

    def build_travel_time_maps(self, detectorpath, station_id, channels_to_include, z_range, r_max, outpath):
        #det = detector.Detector(source="rnog_mongo")
        #det.update(datetime.datetime(2022, 10, 1))
        
        
        det = rnog_detector.Detector(detector_file = detectorpath, log_level=logging.DEBUG)
        det.update(datetime.datetime(2024, 3, 1))
        
        channel_positions = get_channel_positions(det, station_id, channels_to_include)

        z_range_map = (z_range[0] - 1, z_range[1] + 1)
        r_max_map = r_max + 1
        

        for channel, xyz in channel_positions.items():
            print(channel, "channel")

            ttc = TravelTimeCalculator(tx_z = xyz[2],
                                   z_range = z_range_map,
                                   r_max = r_max_map,
                                   num_pts_z = 5 * self.num_pts_z,
                                   num_pts_r = 5 * self.num_pts_r, 
                                   channel = channel)
            
            ttc.set_ior_and_solve(self.ior_model)
            
            


