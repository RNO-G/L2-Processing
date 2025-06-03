from detector import get_channel_positions,calculate_avg_antenna_xyz,get_cable_delays, get_device_position   
#from NuRadioReco.detector.detector import Detector
from NuRadioReco.detector.RNO_G import rnog_detector
import datetime
#from detector import Detector 
from propagation import TravelTimeCalculator
import numpy as np
import utils, reco_utils, preprocessing
import defs
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#import autocolor
import surface_corr 
import summit  
from matplotlib.colors import TwoSlopeNorm
import pickle
import logging 
import datetime as dt
import os 
#from detector import Detector 
from scipy.interpolate import RegularGridInterpolator
import time 
from reco_utils import CorrScoreProvider
import itertools 

class Reco:

    def __init__(self):
        self.num_pts_z = 5000
        self.num_pts_r = 5000
        self.ior_model = defs.ior_exp3
         
     
    def build_travel_time_maps(self, detectorpath, station_id, channels_to_include, z_range, r_max, outpath):
        det = rnog_detector.Detector(detector_file = detectorpath)
        det.update(datetime.datetime(2024, 3, 1))
        
        channel_positions = get_channel_positions(det, station_id, channels_to_include)

        z_range_map = (z_range[0] - 1, z_range[1] + 1)
        r_max_map = r_max + 1

        mapdata = {}
        for channel, xyz in channel_positions.items():
            ttc = TravelTimeCalculator(tx_z = xyz[2],
                                   z_range = z_range_map,
                                   r_max = r_max_map,
                                   num_pts_z = 6 * self.num_pts_z,
                                   num_pts_r = 6 * self.num_pts_r)
        
            ttc.set_ior_and_solve(self.ior_model)
        
            mapdata[channel] = ttc.to_dict()

        with open(outpath, 'wb') as outfile:
            pickle.dump(mapdata, outfile)

        return mapdata


    def run(self, event, station, detectorpath, station_id, channels_to_include, do_envelope, res, ttcs2, run_no):
        
        channel_signals = {}
        channel_times = {}
        for ch in station.iter_channels():
            trace = ch.get_trace()
            times = ch.get_times()
            channel_signals[ch.get_id()] = trace 
            channel_times[ch.get_id()] = times
        
        if do_envelope:
            channel_signals = preprocessing.envelope(channel_signals)
    
        det = rnog_detector.Detector(always_query_entire_description=True,detector_file = detectorpath)
        det.update(datetime.datetime(2024, 3, 1))
        
        channel_pairs_to_include = list(itertools.combinations(channels_to_include, 2))
        csp = CorrScoreProvider(channel_signals, channel_times, channel_pairs_to_include)

        channel_positions = get_channel_positions(det, station_id = station_id, channels = channels_to_include)
        cable_delays = get_cable_delays(det, station_id = station_id, channels = channels_to_include)

        azimuth_range = (-np.pi, np.pi)
        elevation_range = (-np.pi/2, np.pi/2)
        
        res_az = int(2*np.pi/np.deg2rad(1))
        res_elev = int(np.pi/np.deg2rad(1))
        
        elevation_vals = np.linspace(*elevation_range, res_az)
        azimuth_vals = np.linspace(*azimuth_range, res_az)
        ee, aa = np.meshgrid(elevation_vals, azimuth_vals)
        
        #select radius based on data, cal pulser, or simulated neutrinos

        #radius = 300 / defs.cvac 
        radius = 117
        #radius = radi 
        
        origin_xyz = channel_positions[0]  # use PA CH0- as origin of the coordinate system
        
        reco, score, t_ab  = reco_utils.interferometric_reco_ang(ttcs2, channel_signals, channel_times,
                                               rad = radius, origin_xyz = origin_xyz, elevation_range = elevation_range, azimuth_range = azimuth_range,
                                               num_pts_elevation = res_az, num_pts_azimuth = res_az, channels_to_include = channels_to_include,
                                               channel_positions = channel_positions, cable_delays = cable_delays, csp = csp)

        maxcorr_point, maxcorr = utils.get_maxcorr_point(reco)

        surf_corr = surface_corr.SurfaceCorr()

        
        if (True == False):
            fs = 13
            plot_axes = []
            plot_axes_ind = []
            intmap = reco

            for ind, axis in enumerate(["elevation", "azimuth", "radius"]):
                if (axis != "radius"):
                    if len(intmap[axis]) > 1:
                        plot_axes.append(axis)
                        plot_axes_ind.append(ind)
                else:
                    slice_axis = axis
                    slice_val = intmap[slice_axis]

            if len(plot_axes) != 2:
                raise RuntimeError("Error: can only plot 2d maps!")

            axis_a, axis_b = plot_axes
            intmap_to_plot = np.squeeze(intmap["map"])


            figsize = (4.6, 4)
            aspect = "auto"

            fig = plt.figure(figsize = figsize, layout = "constrained")
            gs = GridSpec(1, 1, figure = fig)
            ax = fig.add_subplot(gs[0])

            cscale = np.max(np.abs(intmap["map"]))
            vmax = cscale
            vmin = -cscale
             
            
            im = ax.imshow(np.flip(np.transpose(intmap_to_plot),axis=0),
                    extent = [intmap[axis_b][0], intmap[axis_b][-1],
                             intmap[axis_a][0], intmap[axis_a][-1]],
                   cmap = 'viridis', norm=TwoSlopeNorm(vmin = vmin, vcenter= (vmin + vmax)/2, vmax=vmax), aspect = aspect, interpolation = "none")
            
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
            cbar.ax.tick_params(labelsize = fs)
            cbar.set_label("Correlation", fontsize = fs)
            
            
            pulser_pos = get_device_position(det, station_id = station_id, device_id = 1)
            pulser_r = np.sqrt((pulser_pos[0] - origin_xyz[0])**2 + (pulser_pos[1] - origin_xyz[1])**2)
            pulser_az = np.arctan2(pulser_pos[1] - origin_xyz[1], pulser_pos[0] - origin_xyz[0])
            pulser_elev = np.arctan2(pulser_pos[2] - origin_xyz[2], pulser_r)
            
            ax.scatter(pulser_az, pulser_elev, color = "blue", marker = "*", label = "pulser")

            ax.set_xlabel(f"{axis_b} [m]", fontsize = fs)
            ax.set_ylabel(f"{axis_a} [m]", fontsize = fs)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi/2, np.pi/2)
            ax.tick_params(axis = "y", direction = "in", left = True, right = True, labelsize = fs)
            ax.tick_params(axis = "x", direction = "in", bottom = True, top = True, labelsize = fs)

            ax.text(0.05, 0.92, f"{slice_axis} = {slice_val * defs.cvac:.1f} m", transform = ax.transAxes, fontsize = fs)
            
            ax.axhline(np.arctan2(np.abs(origin_xyz[2]), radius), ls = "dashed", color = "gray")
            ax.scatter(maxcorr_point[plot_axes[1]], maxcorr_point[plot_axes[0]], color = "white", marker = "*", label = "max corr")
            
            ax.legend()
            evt_id = event.get_id()

            if (os.path.exists(f"/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/cp_test_0602/{run_no}") == False):
                os.makedirs(f"/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/cp_test_0602/{run_no}")

            fig.savefig(f"/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/cp_test_0602/{run_no}/{evt_id}_ang_corr.png", dpi = 300)
 
        if (True == False):
            fs = 13
    
            z_range = (-1500, 500)
            r_max = 1500
            r_range = (0,r_max)
            az_reco = maxcorr_point["azimuth"]



            reco, score, t_ab = reco_utils.interferometric_reco_ang2(ttcs2, channel_signals, channel_times, azimuth = az_reco, origin_xyz = origin_xyz, 
                    z_range = z_range, r_range = r_range, num_pts_z = res, num_pts_r = res, channels_to_include = channels_to_include, channel_positions = channel_positions, cable_delays = cable_delays, csp = csp)
            

            intmap = reco

            mapdata = intmap["map"]
            maxind = np.unravel_index(np.argmax(mapdata), mapdata.shape)
            maxcorr_point = {"z": intmap["z"][maxind[1]],
                     "r": intmap["r"][maxind[0]]}
            maxcorr = mapdata[maxind[0]][maxind[1]]
            
            
            plot_axes = []
            plot_axes_ind = []
            
            
            for ind, axis in enumerate(["z", "r", "azimuth"]):
                if axis != "azimuth":
                    plot_axes.append(axis)
                    plot_axes_ind.append(ind)
                else:
                    slice_axis = axis
                    slice_val = intmap[slice_axis]

            if len(plot_axes) != 2:
                raise RuntimeError("Error: can only plot 2d maps!")
            
            
            axis_a, axis_b = plot_axes    
            intmap_to_plot = np.squeeze(intmap["map"])
            

            figsize = (4.6, 4)
            aspect = "equal"
    
            fig = plt.figure(figsize = figsize, layout = "constrained")
            gs = GridSpec(1, 1, figure = fig)
            ax = fig.add_subplot(gs[0])

            cscale = np.max(np.abs(intmap["map"]))
            vmax = cscale
            vmin = -cscale

            delz = z_range[1] - z_range[0]
            delr = r_range[1] - r_range[0]
            aspect = delr/delz

            
            r_vals = np.logspace(0, np.log10(r_range[1]), res // 2)


            """
            im = ax.imshow(np.flip(np.transpose(intmap_to_plot), axis = 0),
                   extent = [intmap[axis_b][0] * defs.cvac, intmap[axis_b][-1] * defs.cvac,
                             intmap[axis_a][0] * defs.cvac, intmap[axis_a][-1] * defs.cvac],
                   cmap = 'viridis', vmin = vmin, vmax=vmax, aspect = "equal")
            """
            z_vals = np.linspace(z_range[0], z_range[1], res // 2)

            im = ax.pcolormesh(r_vals * defs.cvac, z_vals * defs.cvac, np.transpose(intmap_to_plot), cmap = "viridis", shading = "gouraud")
            
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
            cbar.ax.tick_params(labelsize = fs)
            
            cbar.set_label("Correlation", fontsize = fs)
            
            pulser_pos = get_device_position(det, station_id = station_id, device_id = 1)       
            pulser_r = np.sqrt((pulser_pos[0] - origin_xyz[0])**2 + (pulser_pos[1] - origin_xyz[1])**2) * defs.cvac
            pulser_z = (pulser_pos[2] - origin_xyz[2]) * defs.cvac
            
            #ax.set_xscale("log")
            ax.scatter(pulser_r, pulser_z, color = "red", marker = "*", label = "pulser") 
            ax.set_xlabel(f"{axis_b} [m]", fontsize = fs)
            ax.set_ylabel(f"{axis_a} [m]", fontsize = fs)
            ax.tick_params(axis = "y", direction = "in", left = True, right = True, labelsize = fs)
            ax.tick_params(axis = "x", direction = "in", bottom = True, top = True, labelsize = fs)
            #ax.axis("equal")
            ax.text(0.05, 0.92, f"{slice_axis} = {slice_val:.1f}", transform = ax.transAxes, fontsize = fs)
            ax.scatter(maxcorr_point[plot_axes[1]] * defs.cvac, maxcorr_point[plot_axes[0]] * defs.cvac, color = "black", marker = "*", label = "max corr")
            evt_id = event.get_id()
            ax.axhline(np.abs(origin_xyz[2]) * defs.cvac, ls = "dashed", color = "gray")
            ax.legend()
            fig.savefig(f"/data/condor_shared/users/avijai/RNO_reco/rno_dep/source/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/cp_test_0602/{run_no}/{evt_id}_rz_corr.png", dpi = 300)

            
        
        phi_max = maxcorr_point["azimuth"]
        
        z_range = (-1500, 500)
        r_max = 1500
        r_range = (0,r_max)
        
        reco2, score2, t_ab2 = reco_utils.interferometric_reco_ang2(ttcs2, channel_signals, channel_times, azimuth = phi_max, origin_xyz = origin_xyz,
                    z_range = z_range, r_range = r_range, num_pts_z = res, num_pts_r = res, channels_to_include = channels_to_include, channel_positions = channel_positions, cable_delays = cable_delays, csp = csp)

        intmap2 = reco2
        mapdata = intmap2["map"]
        
        maxind = np.unravel_index(np.argmax(mapdata), mapdata.shape)
        maxcorr_point2 = {"z": intmap2["z"][maxind[1]] * defs.cvac,"r": intmap2["r"][maxind[0]] * defs.cvac, "phi" : phi_max, "theta" : maxcorr_point["elevation"]}
        
        maxcorr2 = mapdata[maxind[0]][maxind[1]]
        
        surf_corr_ratio_rz, max_surf_corr_rz, surf_corr_ratio_2_rz, max_surf_corr_2_rz, max_r, max_z, max_z_2, max_r_2 = surf_corr.run(station_id, channels_to_include, reco2, maxcorr2, radius, det)


        return maxcorr_point, maxcorr, score2, t_ab2, surf_corr_ratio_rz, max_surf_corr_rz, surf_corr_ratio_2_rz, max_surf_corr_2_rz, maxcorr_point2, maxcorr2, max_r, max_z, max_z_2, max_r_2 



