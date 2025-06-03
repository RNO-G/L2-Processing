import json, defs
import numpy as np 


def get_channel_positions(det, station_id, channels):
    channel_positions = {}
    for channel in channels:
        channel_positions[channel] = (det.get_relative_position(station_id, channel))/defs.cvac
    return channel_positions

def calculate_avg_antenna_xyz(det, station_id, channels):
    channel_positions = get_channel_positions(det, station_id, channels)
        
    antenna_coordinates = [[] for _ in range(3)]
    for channel in channel_positions:
        for axis in range(3):
            antenna_coordinates[axis].append(channel_positions[channel][axis] * defs.cvac)

    av_ant_position = tuple(np.average(coords) for coords in antenna_coordinates)
    return av_ant_position


def get_cable_delays(det, station_id, channels):
    cable_delays = {}

    for channel in channels:
        cable_delays[channel] = 0

    return cable_delays

def get_device_position(det, station_id, device_id):
    relative_position = det.get_relative_position_device(station_id, device_id)

    return relative_position / defs.cvac


