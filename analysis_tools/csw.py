from reco import Reco
import defs
import utils
import numpy as np
#from detector import Detector
import matplotlib.pyplot as plt
from snr import SNR
import defs 
import dedisperse_new
import reco
import reco_utils
import time
import math
import matplotlib.pyplot as plt 
from NuRadioReco.utilities import units 
from NuRadioReco.detector.RNO_G import rnog_detector
import datetime
from detector import get_channel_positions,calculate_avg_antenna_xyz,get_cable_delays, get_device_position

class CSW:

    def __init__(self):
        self.azimuth_range = (-np.pi, np.pi)
        self.elevation_range = (-np.pi/2, np.pi/2)
        #self.res = 100
        #self.radius = 300 / defs.cvac
        #self.radius = 110 / defs.cvac
        self.zoom_window = 40 * units.ns 

    def get_correlation_function(self, channel_signals, channel_times, channels_to_include, channel_id, reference_ch, solution,  channel_positions, cable_delays, pts, ttcs):

        channel_pairs_to_include = [(channel_id, reference_ch)]
        t_ab, score = reco_utils.calc_corr_score(channel_signals, channel_times, pts, ttcs, channel_pairs_to_include, channel_positions, cable_delays, comps = [solution]) 

        return t_ab, score 

    
    def get_arrival_delays_AraRoot_xcorr(
        self, channel_signals, channel_times, channels_to_include, reference_ch, reco_delays, solution, channel_positions, cable_delays, station_id, ttcs, score, t_ab
    ):

        # Calculate the time delay between each channel and the reference channel
        delays = {}

        origin_xyz = channel_positions[0]
        
        for ch_ID in channels_to_include:

            if ch_ID == reference_ch:
                # Delay between the reference channel and itself will be 0
                delay = 0

            else:

                # Load  the cross correlation for this channel and the reference
                if (ch_ID > reference_ch):
                    channel_pair = (reference_ch, ch_ID)
                    xcorr_times, xcorr_volts = t_ab[channel_pair], score[channel_pair]
                    xcorr_times *= -1
                else:
                    channel_pair = (ch_ID, reference_ch)
                    xcorr_times, xcorr_volts = t_ab[channel_pair], score[channel_pair]
                
                zoomed_indices = np.where(
                    (np.abs( xcorr_times - reco_delays[ch_ID] )) < self.zoom_window // 2
                )[0]


                # Calculate the time of maximum correlation from this
                #   window of expected signal delay.
                if len(zoomed_indices) == 0:
                    delay = xcorr_times[ np.argmax(xcorr_volts) ]
                    
                else:
                    zoomed_indices_filled = np.arange(min(zoomed_indices), max(zoomed_indices), 1)
                    
                
                    delay = xcorr_times[
                        np.argmax(xcorr_volts[zoomed_indices]) # index of max xcorr in zoomed array
                        + zoomed_indices[0] # Adjusted by first zoomed_index
                    ]
                    
                    
            delays[ch_ID] = delay
            
        return delays
    
    def coord_to_frac_pixel(self, coord):
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

    def coord_to_pixel(self, coord):
        return self.coord_to_frac_pixel(coord).astype(int)


    def get_travel_time(self, ttcs, ch, coord):
        pixels = np.transpose(self.coord_to_pixel(coord))
        rows, cols = pixels[0], pixels[1]
        values = (ttcs[ch])[rows, cols]

        return values

    def get_arrival_delays_reco(self, reco_results, channels_to_include, channel_positions, cable_delays, reference_ch, solution, ttcs):

        origin_xyz = channel_positions[0]
        
        z, r = np.meshgrid(reco_results["z"] / defs.cvac, reco_results["r"] / defs.cvac)
        src_pos = utils.ang2_to_cart(z.flatten(), r.flatten(), reco_results["phi"], origin_xyz)
        
        arrival_times = {}

        for ch in channels_to_include:
            arrival_times[ch] = self.get_travel_time(ttcs, ch, utils.to_antenna_rz_coordinates(src_pos, channel_positions[ch]))

        reference_arrival_time = arrival_times[reference_ch]
        arrival_delays = {}
        for ch in channels_to_include:
            arrival_delays[ch] = arrival_times[ch] - reference_arrival_time + cable_delays[ch] - cable_delays[reference_ch]
        
        
        return arrival_delays

    def run(self, event, station, detectorpath, station_id, channels_to_include, solution, ttcs, reco_results, max_corr, score, t_ab):
        
        warning = 0
        
        reco_obj = reco.Reco() 
        
        det = rnog_detector.Detector(detector_file = detectorpath)
        det.update(datetime.datetime(2024, 3, 1))
        channel_positions = get_channel_positions(det, station_id = station_id, channels = list(set([0] + channels_to_include)))
        cable_delays = get_cable_delays(det, station_id = station_id, channels = list(set([0] + channels_to_include)))

        
        channel_times = {}
        channel_signals = {}
        
        for channel in station.iter_channels():
            if (channel.get_id() in channels_to_include):
                volts = channel.get_trace()
                times = channel.get_times()
                channel_times[channel.get_id()] = times
                channel_signals[channel.get_id()] = volts


        channels_to_csw = channels_to_include

        reference_ch = -123456
        reference_ch_max_voltage = -1
        for ch_ID in channels_to_csw:
            this_max_voltage = np.max(channel_signals[ch_ID])
            if this_max_voltage > reference_ch_max_voltage:
                reference_ch_max_voltage = this_max_voltage
                reference_ch = ch_ID
        
        arrival_delays_reco = self.get_arrival_delays_reco(reco_results, channels_to_include, channel_positions, cable_delays, reference_ch, solution, ttcs)
        
    
        arrival_delays = self.get_arrival_delays_AraRoot_xcorr(
        channel_signals, channel_times, channels_to_include, reference_ch, arrival_delays_reco, solution, channel_positions, cable_delays, station_id, ttcs, score, t_ab
        )

        expected_signal_time = np.asarray(channel_times[reference_ch])[
            np.argmax( np.asarray(channel_signals[reference_ch]) )
        ]

        # Initialize the final CSW waveform time and voltage arrays using the
        #   reference channel's time array resized to size of the channel with
        #   the shortest waveform's waveform
        shortest_wf_ch = 123456
        shortest_wf_length = np.inf
        for ch_ID in channels_to_csw:
            if len(channel_signals[ch_ID]) < shortest_wf_length:
                shortest_wf_length = len(channel_signals[ch_ID])
                shortest_wf_ch = ch_ID

        csw_values = np.zeros((1, shortest_wf_length))
        csw_times = np.asarray(
            channel_times[reference_ch])[:shortest_wf_length]
        csw_dt = csw_times[1] - csw_times[0]

        # Roll the waveform from each channel so the starting time of each
        for ch_ID in channels_to_csw:
            values = np.asarray(channel_signals[ch_ID])
            times = np.asarray(channel_times[ch_ID]) - (arrival_delays[ch_ID]//csw_dt)*csw_dt
                
            rebinning_shift = (
                (csw_times[0] - times[0])
                % csw_dt
                # Take the remainder of the start time difference with csw_dt.
                #   If this is ~0, the waveforms have the same binning
                #   otherwise, this reveals how much to shift the waveform by.
                # For example, waveform 1 yeilds: (3.4 - 0.9) % 0.5 = 0
                #   waveform 2 yeilds (1.6 - 1.4) % 0.5 = 0.2
                #   and waveform 3 yeilds (0.9 - 0.5) % 0.5 = 0.4
            )
            if csw_dt - 0.0001 > abs(rebinning_shift) > 0.0001:
                warning += 10_00_00

            # Trim this waveform's length to match the CSW length
            if len(times) > len(csw_times):
                trim_ammount = len(times) - len(csw_times)
                if (
                    ( times[0] - csw_times[0] < 0 ) # this wf has an earlier start time than the CSW
                    and ( times[-1] - csw_times[-1] <= csw_dt/2) # this wf has a earlier or equal end time than the CSW
                ): # We need to trim from the beginning of the waveform
                    times  = times [trim_ammount:]
                    values = values[trim_ammount:]
                elif (
                    ( times[0] - csw_times[0] > -csw_dt/2) # this wf has a later or equal start time than the CSW
                    and (times[-1] - csw_times[-1] > 0) # this wf has a later end time than the CSW
                ): # we need to trim from the end of the waveform
                    times  = times [:-trim_ammount]
                    values = values[:-trim_ammount]
                elif (
                    ( times[0] - csw_times[0] < 0 ) # this wf starts earlier than the CSW
                    and ( times[-1] - csw_times[-1] > 0 ) # this wf ends later than the CSW
                ): # we need to trim from both ends of the waveform
                    leading_trimmable = np.argwhere(
                        np.round(times,5) < np.round(csw_times[0], 5) )
                    trailing_trimmable = np.argwhere(
                        np.round(times, 5) > np.round(csw_times[-1], 5) )
                    times  = times [ len(leading_trimmable) : -len(trailing_trimmable) ]
                    values = values[ len(leading_trimmable) : -len(trailing_trimmable) ]

            roll_shift_bins = (csw_times[0] - times[0]) / csw_dt
            roll_shift_time = roll_shift_bins*(times[1] - times[0])
            if abs(roll_shift_bins) % 1.0 > 0.0001:
                # roll_shift is not close to an integer. Add to the warning
                warning += 10
            roll_shift_bins = int(roll_shift_bins)
            if abs(roll_shift_bins)>len(times):
                # More waveform to roll than there is time in the waveform,
                #   so add to the warning tracker. 
                # Software triggers are so short, this sometimes occurs for them.
                #   Don't warn in this scenario.
                warning += 10_00_00_00
            if roll_shift_bins > 0 and abs(roll_shift_bins)<len(times):
                # Rolling from front to back, check that signal region isn't in the front
                if times[0] <= expected_signal_time <= times[roll_shift_bins]:
                    warning += 10_00
            elif roll_shift_bins < 0 and abs(roll_shift_bins)<len(times):
                # Rolling from back to front, check that signal region isn't in the back
                if  times[roll_shift_bins]  <= expected_signal_time <= times[-1]:
                    warning += 10_00
            rolled_wf = np.roll( values, -roll_shift_bins )
            rolled_times = np.linspace(
                times[0] + roll_shift_time,
                times[-1] + roll_shift_time,
                len(times)
            )
            
            # Add this channel's waveform to the CSW
            csw_values = np.sum( np.dstack( (csw_values[0], rolled_wf) ), axis=2)
        csw_values = np.squeeze(csw_values)
        
        return (csw_times, csw_values)



