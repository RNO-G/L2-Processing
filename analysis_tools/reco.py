from detector import Detector 
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
class Reco:

    def __init__(self):
        #self.z_range = (-2000, 100)
        #self.r_max = 2000
        self.z_range = (-2000, 300)
        self.r_max = 2000
        self.num_pts_z = 100
        self.num_pts_r = 100
        self.ior_model = defs.ior_exp3
         

    
    def build_travel_time_maps(self, detectorpath, station_id, channels_to_include):

        det = Detector(detectorpath)
        channel_positions = det.get_channel_positions(station_id, channels_to_include)

        z_range_map = (self.z_range[0] - 1, self.z_range[1] + 1)
        r_max_map = self.r_max + 1

        mapdata = {}
        for channel, xyz in channel_positions.items():
            ttc = TravelTimeCalculator(tx_z = xyz[2],
                                   z_range = z_range_map,
                                   r_max = r_max_map,
                                   num_pts_z = 5 * self.num_pts_z,
                                   num_pts_r = 5 * self.num_pts_r)
        
            ttc.set_ior_and_solve(self.ior_model)

            mapdata[channel] = ttc.to_dict()

        return mapdata

    def run(self, event, station, detectorpath, station_id, channels_to_include, do_envelope, res, mappath, ttcs):
        
        #mappath = self.build_travel_time_maps(detectorpath, station_id, channels_to_include)
        
        channel_signals = {}
        channel_times = {}
        for ch in station.iter_channels():
            trace = ch.get_trace()
            times = ch.get_times()
            channel_signals[ch.get_id()] = trace 
            channel_times[ch.get_id()] = times
        
        if do_envelope:
            channel_signals = preprocessing.envelope(channel_signals)

        det = Detector(detectorpath)
        channel_positions = det.get_channel_positions(station_id = station_id, channels = channels_to_include)
        cable_delays = det.get_cable_delays(station_id = station_id, channels = channels_to_include)
        
        azimuth_range = (-np.pi, np.pi)
        elevation_range = (-np.pi/2, np.pi/2)

        elevation_vals = np.linspace(*elevation_range, res)
        azimuth_vals = np.linspace(*azimuth_range, res)
        ee, aa = np.meshgrid(elevation_vals, azimuth_vals)

        radius = 90 / defs.cvac

        origin_xyz = channel_positions[0]  # use PA CH0- as origin of the coordinate system
        #origin_xyz = [0,0,0]
        #ttcs = utils.load_ttcs(mappath, channels_to_include)


        reco, score, t_ab = reco_utils.interferometric_reco_ang(ttcs, channel_signals, channel_times, mappath,
                                               rad = radius, origin_xyz = origin_xyz, elevation_range = elevation_range, azimuth_range = azimuth_range,
                                               num_pts_elevation = res, num_pts_azimuth = res, channels_to_include = channels_to_include,
                                               channel_positions = channel_positions, cable_delays = cable_delays)

        maxcorr_point, maxcorr = utils.get_maxcorr_point(reco)
        surf_corr = surface_corr.SurfaceCorr()

        surf_corr_ratio, max_surf_corr = surf_corr.run(station_id, channels_to_include, reco, maxcorr, radius, det)
        """
        summ = summit.Summit()
        ((x_summ, y_summ, z_summ), (x_disc, y_disc, z_disc)) = summ.run(station_id)
        
        if (event.get_id() == 2626):
            fs = 13
            # Figure out how to plot this map:
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

            #if autocolor:
                #vmin = None
                #vmax = None
            #else:
            cscale = np.max(np.abs(intmap["map"]))
            vmax = cscale
            vmin = -cscale

            im = ax.imshow(np.flip(np.transpose(intmap_to_plot), axis = 0),
                    extent = [intmap[axis_b][0], intmap[axis_b][-1],
                             intmap[axis_a][0], intmap[axis_a][-1]],
                   cmap = "bwr", vmax = vmax, vmin = vmin, aspect = aspect)

            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
            cbar.ax.tick_params(labelsize = fs)
            cbar.set_label("Correlation", fontsize = fs)

            pulser_pos = det.get_device_position(station_id = station_id, devices = [1])
            
            pulser_r = np.sqrt((pulser_pos[1][0] - origin_xyz[0])**2 + (pulser_pos[1][1] - origin_xyz[1])**2)
            pulser_az = np.arctan2(pulser_pos[1][1] - origin_xyz[1], pulser_pos[1][0] - origin_xyz[0])
            pulser_elev = np.arctan2(pulser_pos[1][2] - origin_xyz[2], pulser_r)
            
            summ_r = np.sqrt(x_summ**2 + y_summ**2)
            summ_az = np.arctan2(y_summ, x_summ)
            summ_elev = np.arctan2(z_summ/defs.cvac - origin_xyz[2], summ_r)
            
            disc_r = np.sqrt(x_disc**2 + y_disc**2)
            disc_az = np.arctan2(y_disc, x_disc)
            disc_elev = np.arctan2(z_disc/defs.cvac - origin_xyz[2], disc_r)

            ax.scatter(summ_az, summ_elev, color = "blue", marker = "*", label = "summit")
            ax.scatter(disc_az, disc_elev, color = "black", marker = "*", label = "disc")
            #ax.scatter(pulser_az, pulser_elev, color = "blue", marker = "*", label = "detector")

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
            fig.savefig("683_2626_map_ang_orig.png", dpi = 300)

 
        if (event.get_id() == 2626):
            fs = 13
    
            # pick some reasonable domain
            z_range = (-2000, 100)
            r_max = 1000
            r_range = (0,r_max)

            az_reco = maxcorr_point["azimuth"]
            origin_xyz = channel_positions[0]
            #origin_xyz = [0,0,0]


            ttcs = utils.load_ttcs(mappath, channels_to_include)

            reco, score, t_ab = reco_utils.interferometric_reco_ang2(ttcs, channel_signals, channel_times, mappath, azimuth = az_reco, origin_xyz = origin_xyz, 
                    z_range = z_range, r_range = r_range, num_pts_z = res, num_pts_r = res, channels_to_include = channels_to_include, channel_positions = channel_positions, cable_delays = cable_delays)

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
            
            im = ax.imshow(np.flip(np.transpose(intmap_to_plot), axis = 0),
                   extent = [intmap[axis_b][0] * defs.cvac, intmap[axis_b][-1] * defs.cvac,
                             intmap[axis_a][0] * defs.cvac, intmap[axis_a][-1] * defs.cvac],
                   cmap = "viridis", vmin = vmin, vmax = vmax, aspect = aspect)
            
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
            cbar.ax.tick_params(labelsize = fs)
            cbar.set_label("Correlation", fontsize = fs)
            pulser_pos = det.get_device_position(station_id = station_id, devices = [1])       
            pulser_r = np.sqrt((pulser_pos[1][0] - origin_xyz[0])**2 + (pulser_pos[1][1] - origin_xyz[1])**2) * defs.cvac 
            pulser_z = (pulser_pos[1][2] - origin_xyz[2]) * defs.cvac
            
            summ_r = np.sqrt(x_summ**2 + y_summ**2)
            summ_z = z_summ - (origin_xyz[2] * defs.cvac)

            disc_r = np.sqrt(x_disc**2 + y_disc**2)
            disc_z = z_disc - (origin_xyz[2] * defs.cvac)
            
            ax.scatter(summ_r, summ_z, color = "red", marker = "*", label = "summit")
            ax.scatter(disc_r, disc_z, color = "blue", marker = "*", label = "disc")

            #ax.scatter(pulser_r, pulser_z, color = "red", marker = "*", label = "detector") 
            ax.set_xlabel(f"{axis_b} [m]", fontsize = fs)
            ax.set_ylabel(f"{axis_a} [m]", fontsize = fs)
            ax.tick_params(axis = "y", direction = "in", left = True, right = True, labelsize = fs)
            ax.tick_params(axis = "x", direction = "in", bottom = True, top = True, labelsize = fs)
            ax.axis("equal") 
            ax.text(0.05, 0.92, f"{slice_axis} = {slice_val:.1f}", transform = ax.transAxes, fontsize = fs)
            ax.scatter(maxcorr_point[plot_axes[1]] * defs.cvac, maxcorr_point[plot_axes[0]] * defs.cvac, color = "black", marker = "*", label = "max corr")
            ax.legend()
            ax.axhline(np.abs(origin_xyz[2]) * defs.cvac, ls = "dashed", color = "gray")
            fig.savefig("683_2626_map_rz_orig.png", dpi = 300)

            print("done 2626")
            
        

        
        phi_max = maxcorr_point["azimuth"]

        reco2, score2, t_ab2 = reco_utils.interferometric_reco_ang2(ttcs, channel_signals, channel_times, mappath, azimuth = phi_max, origin_xyz = origin_xyz,
                    z_range = (-1000,10), r_range = (0, 1000), num_pts_z = res, num_pts_r = res, channels_to_include = channels_to_include, channel_positions = channel_positions, cable_delays = cable_delays)

        intmap2 = reco2
        mapdata = intmap2["map"]
        
        maxind = np.unravel_index(np.argmax(mapdata), mapdata.shape)
        maxcorr_point2 = {"z": intmap2["z"][maxind[1]] * defs.cvac,"r": intmap2["r"][maxind[0]] * defs.cvac, "phi" : phi_max, "theta" : maxcorr_point["elevation"]}
        maxcorr2 = mapdata[maxind[0]][maxind[1]]

        if (maxcorr2 > maxcorr): 
            print("Higher", maxcorr2, maxcorr)
        else:
            print("Lower or Equal", maxcorr2, maxcorr)
        
        """
        return maxcorr_point, maxcorr, score, t_ab, surf_corr_ratio, max_surf_corr



