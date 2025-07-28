import pykonal, copy
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import gc
import zarr 

class TravelTimeCalculator:

    # All coordinates here are 2d (r, z) in natural feet

    @classmethod
    def FromDict(cls, indict):
        obj = cls(**indict)
        return obj

    def __init__(self, tx_z, z_range, r_max, num_pts_z, num_pts_r, channel, travel_time_maps = {}, r_vals = [], z_vals = []):

        self.tx_z = tx_z
        self.tx_pos = [0.0, self.tx_z]

        self.num_pts_z = num_pts_z
        self.num_pts_r = num_pts_r

        self.z_range = z_range
        self.r_max = r_max

        self.r_vals = np.linspace(0, r_max, num_pts_r)
        self.z_vals = np.linspace(z_range[0], z_range[1], num_pts_z)

        self.channel = channel 

        self.domain_start = np.array([0, self.z_range[0]])
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
    

    def find_turning_points(self, ior_func, z_vals):
        n0 = ior_func(self.tx_z)
        turning_points = []

        n_z = ior_func(z_vals)

        theta_vals = np.linspace(-pi/2, pi/2, 180)

        for theta in theta_vals:
            p = n0 * np.sin(theta)

            diff = np.abs(n_z - p)
            idx = np.argmin(diff)

            if n_z[idx] >= p:
                if 0 < idx < len(z_vals) - 1:
                    slope_before = (n_z[idx] - n_z[idx-1]) / (z_vals[idx] - z_vals[idx-1])
                    slope_after = (n_z[idx+1] - n_z[idx]) / (z_vals[idx+1] - z_vals[idx])

                    if slope_before * slope_after < 0:
                        turning_points.append(z_vals[idx])

        return turning_points

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
        #iorslice = [1.78] * self.num_pts_z
        #turning_points = self.find_turning_points(ior, zvals)
        iordata = np.expand_dims(np.tile(iorslice, reps = (self.num_pts_r, 1)), axis = -1)
        
        boundary_z_ind = self._coord_to_pykonal([[0, reflection_at_z]])[0][1]
        
        """
        #Calculate refracted rays 
        for turning_pt in turning_points:
            solver = _get_solver(iordata)
            src_ind = self._coord_to_pykonal([self.tx_pos])[0]
            solver.traveltime.values[src_ind[0], src_ind[1], src_ind[2]] = 0 # Place a point source at the transmitter
            solver.unknown[src_ind[0], src_ind[1], src_ind[2]] = False
            solver.trial.push(src_ind[0], src_ind[1], src_ind[2])
            solver.solve()
            
            inf_pos = [0, turning_pt]
            inf_ind = self._coord_to_pykonal([inf_pos])[0]
            t1 = solver1.traveltime.values[inf_ind[0], inf_ind[1], inf_ind[2]]

            solver2 = _get_solver(iordata)

            solver2.traveltime.values[inf_ind[0], inf_ind[1], inf_ind[2]] = 0
            solver2.unknown[inf_ind[0], inf_ind[1], inf_ind[2]] = False
            solver2.trial.push(inf_ind[0], inf_ind[1], inf_ind[2])
            solver2.solve()

            recv_ind = self._coord_to_pykonal([receiver_xyz])[0]
            t2 = solver2.traveltime.values[*recv_ind]
        """



        # Calculate rays transmitted into the air
        solver = _get_solver(iordata)
        src_ind = self._coord_to_pykonal([self.tx_pos])[0]

        solver.traveltime.values[src_ind[0], src_ind[1], src_ind[2]] = 0 # Place a point source at the transmitter
        solver.unknown[src_ind[0], src_ind[1], src_ind[2]] = False
        solver.trial.push(src_ind[0], src_ind[1], src_ind[2])
        solver.solve()

        combined_map = np.full_like(solver.traveltime.values, np.nan)
        combined_map[:, boundary_z_ind:, :] = solver.traveltime.values[:, boundary_z_ind:, :]
        del solver
        gc.collect()

        
        # Calculate direct rays in the ice
        iordata[:, boundary_z_ind:, :] = 10.0 # assign a spuriously large IOR to the air to make sure there are no head waves
                                              # that can overtake the bulk-bending modes that we want
        solver = _get_solver(iordata)
        src_ind = self._coord_to_pykonal([self.tx_pos])[0]

        solver.traveltime.values[src_ind[0], src_ind[1], src_ind[2]] = 0 # Place a point source at the transmitter
        solver.unknown[src_ind[0], src_ind[1], src_ind[2]] = False
        solver.trial.push(src_ind[0], src_ind[1], src_ind[2])
        solver.solve()
        
        
        #direct_ice_map = solver.traveltime.values
        #direct_ice_map[:, boundary_z_ind+1:, :] = np.nan
        
        combined_map[:, :boundary_z_ind, :] = solver.traveltime.values[:, :boundary_z_ind, :]
        
        del solver
        gc.collect()
        
        """
        solver = _get_solver(iordata)
        solver.traveltime.values[:, boundary_z_ind, :] = direct_ice_map[:, boundary_z_ind, :]

        solver.unknown[:, boundary_z_ind, :] = False
        for r_ind in range(self.num_pts_r):
            solver.trial.push(r_ind, boundary_z_ind, 0)
        solver.solve()

        combined_map[:, :boundary_z_ind, :] = solver.traveltime.values[:, :boundary_z_ind, :]
        
        del solver
        del direct_ice_map
        gc.collect()
        
        """
        self.travel_time_maps["direct_combined"] = combined_map
        
        
        chunk_shape = (1000,1000)
         
        z = zarr.open(
                f'/home/avijai/scratch/NuRadioMC/NuRadioReco/examples/RNO_data/read_data_example/ttimes_0723_ior1/ttimes_{self.channel}.zarr',# output folder
                mode='w',                      # 'w' to overwrite
                shape=combined_map.shape,      # full array shape
                chunks=chunk_shape,            # chunk size
                dtype='float64',               # retain precision
                compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=1)  # efficient compression)
                )

        z[:] = combined_map 
        

    def get_ind(self, coord):

        return np.transpose(self._coord_to_pixel(coord))

    def get_travel_time(self, coord, comp = "direct_combined"):

        if comp not in self.travel_time_maps:
            raise RuntimeError(f"Error: map for component '{comp}' not available!")

        ind = self.get_ind(coord)
        
        return self.travel_time_maps[comp][src_ind[0], src_ind[1], src_ind[2]]
        #return self.travel_time_maps[comp][*ind]

    def get_travel_time_ind(self, ind, comp = "direct_combined"):
        return self.travel_time_maps[comp][src_ind[0], src_ind[1], src_ind[2]]
        #return self.travel_time_maps[comp][*ind]

    def get_tangent_vector(self, coord, comp = "direct_combined"):

        if comp not in self.travel_time_maps:
            raise RuntimeError(f"Error: map for component '{comp}' not available!")

        ind_r, ind_z, _ = self.get_ind(coord)
        return self.tangent_vectors[comp][ind_r, ind_z]
    
    def get_position_offset(self, coord):
        coord = np.array(coord)  # (r, z)

        # Convert coord to fractional pixel index
        pixel_frac = (coord - self.domain_start) / (self.domain_end - self.domain_start) * self.domain_shape
        pixel_int = pixel_frac.astype(int)

        # Convert back to physical space
        #snapped_coord = self.domain_start + pixel_int / self.domain_shape * (self.domain_end - self.domain_start)
        snapped_coord = (pixel_int / self.domain_shape * (self.domain_end - self.domain_start)) + self.domain_start
        

        offset = snapped_coord - coord
        return offset, snapped_coord 

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
