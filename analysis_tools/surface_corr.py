import defs
import math

class SurfaceCorr:

    def __init__(self):
        self.z_thresh = -10 

    def run(self, station_id, channels_to_include, intmap, maxcorr, radius, det):
        
        radius *= defs.cvac
        _, _, avg_z = det.calculate_avg_antenna_xyz(station_id, channels_to_include)
        
        if radius < (abs(avg_z) + self.z_thresh):
            return -np.inf, 0, 0 # surface not visible, this map has no surface corr max
        
        theta_thresh = math.asin((abs(avg_z) + self.z_thresh) / radius)
        row, col = intmap["map"].shape
        
        max_surf_corr = -2
        max_theta = 0
        max_phi = 0

        for r in range(row):
            for c in range(col):
                if (intmap["elevation"][c] >= theta_thresh):
                    if (intmap["map"][r][c] > max_surf_corr):
                        max_surf_corr = intmap["map"][r][c]
                        max_theta = intmap["elevation"][c]
                        max_phi = intmap["azimuth"][r]

        if (maxcorr != 0):
            surf_corr_ratio = max_surf_corr / maxcorr 
        else:
            surf_corr_ratio = np.inf 

        return surf_corr_ratio, max_surf_corr 

    


