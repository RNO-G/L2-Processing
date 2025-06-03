import utils, pickle, itertools, math
import numpy as np, scipy.signal as signal
from propagation import TravelTimeCalculator
import math
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.signal import hilbert
import dask.array as da
import gc
import time 


class CorrScoreProvider:
    
    def __init__(self, channel_sigvals, channel_times, channel_pairs_to_include, upsample = 10):

        self.corrs = {}
        self.tvals = {}

        for (ch_a, ch_b) in channel_pairs_to_include:
            tvals_a, tvals_b = channel_times[ch_a], channel_times[ch_b]
            sig_a, sig_b = channel_sigvals[ch_a], channel_sigvals[ch_b]
            

            # upsample both signals onto a common fine grid
            
            target_dt = min(tvals_a[1] - tvals_a[0], tvals_b[1] - tvals_b[0]) / upsample

            sig_a_tvals_rs, sig_a_rs = utils.resample(tvals_a, sig_a, target_dt)
            sig_b_tvals_rs, sig_b_rs = utils.resample(tvals_b, sig_b, target_dt)
            

            sig_a_rs_norm = (sig_a_rs - np.mean(sig_a_rs)) / np.std(sig_a_rs)
            sig_b_rs_norm = (sig_b_rs - np.mean(sig_b_rs)) / np.std(sig_b_rs)
            
            sig_a_rs_norm_window = sig_a_rs_norm
            
            sig_b_rs_norm_window = sig_b_rs_norm

            normfact = signal.correlate(np.ones(len(sig_a_rs_norm_window)), np.ones(len(sig_b_rs_norm_window)), mode = "full")
            corrs = signal.correlate(sig_a_rs_norm_window, sig_b_rs_norm_window, mode = "full") / normfact
            lags = signal.correlation_lags(len(sig_a_rs_norm_window), len(sig_b_rs_norm_window), mode = "full")
            
            tvals = lags * target_dt + sig_a_tvals_rs[0] - sig_b_tvals_rs[0]
            
            from scipy.signal import windows
            windows3 = windows.hann(len(corrs))
            import scipy 
            corrs *= windows3

            self.corrs[(ch_a, ch_b)] = corrs
            self.tvals[(ch_a, ch_b)] = tvals

    def get(self, ch_a, ch_b, t_ab):
        corrvals = self.corrs[(ch_a, ch_b)]
        tvals = self.tvals[(ch_a, ch_b)]

        return np.interp(t_ab, tvals, corrvals, left=0, right=0)

def coord_to_frac_pixel(coord):
    if isinstance(coord, list):
        coord = np.array(coord)
        
    z_range = (-7000, 2000)
    r_max = 7000
    num_pts_z = 30000 
    num_pts_r = 30000
        
    domain_start = np.array([0.0, z_range[0]])
    domain_end = np.array([r_max, z_range[1]])
    domain_shape = np.array([num_pts_r, num_pts_z])


    pixel_2d = (coord - domain_start) / (domain_end - domain_start) * domain_shape
    pixel_3d = np.append(pixel_2d, np.zeros((len(coord), 1)), axis = 1)

    return pixel_2d

def coord_to_pixel(coord):
    return coord_to_frac_pixel(coord).astype(int)

def get_travel_time(ttcs, ch, coord):
    pixels = np.transpose(coord_to_pixel(coord)) 
    rows, cols = pixels[0], pixels[1]
    
    values = (ttcs[ch])[rows, cols]

    return values

def calc_corr_score(channel_signals, channel_times, pts, ttcs, origin, channel_pairs_to_include, channel_positions, cable_delays,
                    csp, comps = ["direct_combined"]):
    
    corrs = csp.corrs
    tvals = csp.tvals
    

    channels = list(set([channel for pair in channel_pairs_to_include for channel in pair]))

    travel_times = {ch: get_travel_time(ttcs, ch, (utils.to_antenna_rz_coordinates(pts, channel_positions[ch]))) for ch in channels}

    scores = []
    t_abs = []


    for (ch_a, ch_b) in channel_pairs_to_include:
        sig_a = channel_signals[ch_a]
        sig_b = channel_signals[ch_b]
        tvals_a = channel_times[ch_a]
        tvals_b = channel_times[ch_b]
        

        for comp in comps:
            t_ab = travel_times[ch_a] - travel_times[ch_b]
            
            score = csp.get(ch_a, ch_b, t_ab)

            scores.append(np.nan_to_num(score, nan = 0.0))
    
    return tvals, np.mean(scores, axis = 0), corrs



def build_interferometric_map_3d(channel_signals, channel_times, channel_pairs_to_include, channel_positions, cable_delays,
                                 coord_start, coord_end, num_pts, ttcs):

    x_vals = np.linspace(coord_start[0], coord_end[0], num_pts[0])
    y_vals = np.linspace(coord_start[1], coord_end[1], num_pts[1])
    z_vals = np.linspace(coord_start[2], coord_end[2], num_pts[2])
    
    xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing = 'ij')
    pts = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis = -1)

    t_ab, intmap, scores_all = calc_corr_score(channel_signals, channel_times, pts, ttcs, [0,0,0], channel_pairs_to_include,
                             channel_positions = channel_positions, cable_delays = cable_delays)
    assert len(intmap) == len(pts)
    intmap = np.reshape(intmap, num_pts, order = "C")

    return x_vals, y_vals, z_vals, intmap, scores_all, t_ab

# all coordinates and coordinate ranges are given in natural feet
def interferometric_reco_3d(ttcs, channel_signals, channel_times,
                            coord_start, coord_end, num_pts,
                            channels_to_include, channel_positions, cable_delays):

    channel_pairs_to_include = list(itertools.combinations(channels_to_include, 2))
    x_vals, y_vals, z_vals, intmap, score, t_ab = build_interferometric_map_3d(channel_signals, channel_times, channel_pairs_to_include,
                                                                  channel_positions = channel_positions, cable_delays = cable_delays,
                                                                  coord_start = coord_start, coord_end = coord_end, num_pts = num_pts,
                                                                  ttcs = ttcs)
       
    reco_event = {
        "x": x_vals,
        "y": y_vals,
        "z": z_vals,
        "map": intmap
    }
    
    return reco_event, score, t_ab

def build_interferometric_map_ang(channel_signals, channel_times, channel_pairs_to_include, channel_positions, cable_delays,
                                  rad, origin_xyz, elevation_range, azimuth_range, num_pts_elevation, num_pts_azimuth, ttcs, csp):
    
    elevation_vals = np.linspace(*elevation_range, num_pts_elevation)
    azimuth_vals = np.linspace(*azimuth_range, num_pts_azimuth)
    

    ee, aa = np.meshgrid(elevation_vals, azimuth_vals)


    # convert to cartesian points
    pts = utils.ang_to_cart(ee.flatten(), aa.flatten(), radius = rad, origin_xyz = origin_xyz)


    t_ab, intmap, scores_all  = calc_corr_score(channel_signals, channel_times, pts, ttcs, origin_xyz, channel_pairs_to_include, 
                             channel_positions = channel_positions, cable_delays = cable_delays, csp = csp, comps = ["direct_combined"])

    
    intmap = np.reshape(intmap, (num_pts_elevation, num_pts_azimuth), order = "C")
    return elevation_vals, azimuth_vals, intmap, scores_all, t_ab


def build_interferometric_map_ang2(channel_signals, channel_times, channel_pairs_to_include, channel_positions, cable_delays,
                                  azimuth, origin_xyz, z_range, r_range, num_pts_z, num_pts_r, ttcs, csp):
    
    z_vals = np.linspace(*z_range, num_pts_z // 2)
    r_vals = np.logspace(0, np.log10(r_range[1]), num_pts_r // 2)
    zz, rr = np.meshgrid(z_vals, r_vals)

    # convert to cartesian points
    pts = utils.ang2_to_cart(zz.flatten(), rr.flatten(), azimuth = azimuth, origin_xyz = origin_xyz)
    
    t_ab, intmap, scores_all = calc_corr_score(channel_signals, channel_times, pts, ttcs, origin_xyz, channel_pairs_to_include,
                             channel_positions = channel_positions, cable_delays = cable_delays, csp = csp, comps = ["direct_combined"])


    intmap = np.reshape(intmap, (num_pts_z // 2, num_pts_r // 2), order = "C")

    return z_vals, r_vals, intmap, scores_all, t_ab


def interferometric_reco_ang(ttcs, channel_signals, channel_times,
                             rad, origin_xyz, elevation_range, azimuth_range, num_pts_elevation, num_pts_azimuth,
                             channels_to_include, channel_positions, cable_delays, csp):

    channel_pairs_to_include = list(itertools.combinations(channels_to_include, 2))
    elevation_vals, azimuth_vals, intmap, score, t_ab = build_interferometric_map_ang(channel_signals, channel_times, channel_pairs_to_include,
                                                                         channel_positions = channel_positions, cable_delays = cable_delays,
                                                                         rad = rad, origin_xyz = origin_xyz, elevation_range = elevation_range, azimuth_range = azimuth_range,
                                                                         num_pts_elevation = num_pts_elevation, num_pts_azimuth = num_pts_azimuth, ttcs = ttcs, csp = csp)

    reco_event = {
        "elevation": elevation_vals,
        "azimuth": azimuth_vals,
        "radius": rad,
        "map": intmap
    }
    
    return reco_event, score, t_ab

def interferometric_reco_ang2(ttcs, channel_signals, channel_times,
                             azimuth, origin_xyz, z_range, r_range, num_pts_z, num_pts_r,
                             channels_to_include, channel_positions, cable_delays, csp):

    channel_pairs_to_include = list(itertools.combinations(channels_to_include, 2))
    z_vals, r_vals, intmap, score, t_ab = build_interferometric_map_ang2(channel_signals, channel_times, channel_pairs_to_include,
                                                                         channel_positions = channel_positions, cable_delays = cable_delays,
                                                                         azimuth = azimuth, origin_xyz = origin_xyz, z_range = z_range, r_range = r_range,
                                                                         num_pts_z = num_pts_z, num_pts_r = num_pts_r, ttcs = ttcs, csp = csp)

    reco_event = {
        "z": z_vals,
        "r": r_vals,
        "azimuth": azimuth,
        "map": intmap
    }

    return reco_event, score, t_ab

