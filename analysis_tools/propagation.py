import pykonal, copy
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import gc 
import dask.array as da

class TravelTimeCalculator:

    # All coordinates here are 2d (r, z) in natural feet

    @classmethod
    def FromDict(cls, indict):
        print(indict["r_max"], indict['z_range'], indict['num_pts_z'], indict['num_pts_r'], indict["tx_z"])

        obj = cls(**indict)
        return obj
    
    def __init__(self, tx_z, z_range, r_max, num_pts_z, num_pts_r, travel_time_maps = {}, r_vals = [], z_vals = []):

        self.tx_z = tx_z
        self.tx_pos = [0.0, self.tx_z]
        
        self.num_pts_z = num_pts_z
        self.num_pts_r = num_pts_r
        
        self.z_range = z_range
        self.r_max = r_max

        self.r_vals = np.linspace(0, r_max, num_pts_r)
        self.z_vals = np.linspace(z_range[0], z_range[1], num_pts_z)
        
        
        self.domain_start = np.array([0.0, self.z_range[0]])
        self.domain_end = np.array([self.r_max, self.z_range[1]])
        self.domain_shape = np.array([self.num_pts_r, self.num_pts_z])    

        # determine voxel size
        self.delta_r = self.r_max / self.num_pts_r

        self.delta_z = (self.z_range[1] - self.z_range[0]) / self.num_pts_z
        
        self.travel_time_maps = travel_time_maps


    def to_dict(self):        
        return {
            "tx_z": self.tx_z,
            "z_range": self.z_range,
            "r_max": self.r_max,
            "num_pts_z": self.num_pts_z,
            "num_pts_r": self.num_pts_r,
            "travel_time_maps": self.travel_time_maps
        }
    
    
    def set_ior_and_solve(self, ior, reflection_at_z = 0.0):

        def _get_solver(iordata):
            veldata = 1.0 / iordata  # c = 1 when distance measured in natural feet
            solver = pykonal.EikonalSolver(coord_sys = "cartesian")
            solver.velocity.min_coords = 0, 0, 0
            solver.velocity.npts = self.num_pts_r, self.num_pts_z, 1
            solver.velocity.node_intervals = self.delta_r, self.delta_z, 1        
            solver.velocity.values = veldata
            return solver
        
        # Build the IOR distribution
        zvals = np.linspace(self.z_range[0], self.z_range[1], self.num_pts_z)
        iorslice = ior(zvals)
        iordata = np.expand_dims(np.tile(iorslice, reps = (self.num_pts_r, 1)), axis = -1)
        
        boundary_z_ind = self._coord_to_pykonal([[0, reflection_at_z]])[0][1]

        # Calculate rays transmitted into the air
        solver = _get_solver(iordata)
        src_ind = self._coord_to_pykonal([self.tx_pos])[0]
        
        solver.traveltime.values[*src_ind] = 0 # Place a point source at the transmitter
        solver.unknown[*src_ind] = False    
        solver.trial.push(*src_ind)
        solver.solve()
        
        combined_map = np.full_like(solver.traveltime.values.astype(np.float32), np.nan)
        combined_map[:, boundary_z_ind:, :] = solver.traveltime.values.astype(np.float32)[:, boundary_z_ind:, :]
        del solver
        gc.collect()

        # Calculate direct rays in the ice
        iordata[:, boundary_z_ind:, :] = 10.0 # assign a spuriously large IOR to the air to make sure there are no head waves
                                              # that can overtake the bulk-bending modes that we want
        solver = _get_solver(iordata)
        src_ind = self._coord_to_pykonal([self.tx_pos])[0]
        
        solver.traveltime.values[*src_ind] = 0 # Place a point source at the transmitter
        solver.unknown[*src_ind] = False    
        solver.trial.push(*src_ind)
        solver.solve()

        
        combined_map[:, :boundary_z_ind, :] = solver.traveltime.values.astype(np.float32)[:, :boundary_z_ind, :]
        del solver
        gc.collect()
        
        self.travel_time_maps["direct_combined"] = combined_map


    def get_ind(self, coord): 
        return np.transpose(self._coord_to_pixel(coord))        
        
    def get_travel_time(self, coord, comp = "direct_combined"):

        if comp not in self.travel_time_maps:
            raise RuntimeError(f"Error: map for component '{comp}' not available!")

        ind = self.get_ind(coord)

        return self.travel_time_maps[comp][*ind]

    def get_travel_time_ind(self, ind, comp = "direct_combined"):
        return self.travel_time_maps[comp][*ind]
    
    def get_tangent_vector(self, coord, comp = "direct_combined"):

        if comp not in self.travel_time_maps:
            raise RuntimeError(f"Error: map for component '{comp}' not available!")

        ind_r, ind_z, _ = self.get_ind(coord)
        return self.tangent_vectors[comp][ind_r, ind_z]
    
    def _coord_to_pykonal(self, coord):
        return tuple(self._coord_to_pixel(coord))
        
    def _coord_to_pixel(self, coord):
        return self._coord_to_frac_pixel(coord).astype(int)
    
    
    def _coord_to_frac_pixel(self, coord):        
        if isinstance(coord, list):
            coord = np.array(coord)
        
        
        pixel_2d = (coord - self.domain_start) / (self.domain_end - self.domain_start) * self.domain_shape
        pixel_3d = np.append(pixel_2d, np.zeros((len(coord), 1)), axis = 1)
        
        return pixel_3d
