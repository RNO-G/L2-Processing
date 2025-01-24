import json, defs
import numpy as np 

class Detector:

    def __init__(self, json_path, var = None):
        with open(json_path, 'r') as infile:
            if var is not None:
                self.data = json.load(infile)[var]
            else:
                self.data = json.load(infile)

    def get_channel_positions(self, station_id, channels):
        channel_positions = {}

        for channel in channels:
            for entry_id, entry in self.data["channels"].items():
                if entry["station_id"] == station_id and entry["channel_id"] == channel:
                    channel_positions[channel] = [entry["ant_position_x"] / defs.cvac, entry["ant_position_y"] / defs.cvac, entry["ant_position_z"] / defs.cvac]

        return channel_positions

    def calculate_avg_antenna_xyz(self, station_id, channels):
        channel_positions = self.get_channel_positions(station_id, channels)
        
        antenna_coordinates = [[] for _ in range(3)]
        for channel in channel_positions:
            for axis in range(3):
                antenna_coordinates[axis].append(channel_positions[channel][axis] * defs.cvac)

        av_ant_position = tuple(np.average(coords) for coords in antenna_coordinates)
        return av_ant_position



    def get_cable_delays(self, station_id, channels):
        cable_delays = {}

        for channel in channels:
            for entry_id, entry in self.data["channels"].items():
                if entry["station_id"] == station_id and entry["channel_id"] == channel:
                    cable_delays[channel] = entry["cab_time_delay"]

        return cable_delays

    def get_device_position(self, station_id, devices):
        device_positions = {}

        for device in devices:
            for entry_id, entry in self.data["devices"].items():
                if entry["station_id"] == station_id and entry["device_id"] == device:
                    device_positions[device] = [entry["ant_position_x"] / defs.cvac, entry["ant_position_y"] / defs.cvac, entry["ant_position_z"] / defs.cvac]

        return device_positions

    
