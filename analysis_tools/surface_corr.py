import defs
import math
import time
import numpy as np
from detector import get_channel_positions,calculate_avg_antenna_xyz

class SurfaceCorr:

    def __init__(self):
        self.z_thresh = -10 

    def run(self, station_id, channels_to_include, intmap, maxcorr, radius, det):
        
        radius *= defs.cvac

        
        _, _, avg_z = calculate_avg_antenna_xyz(det, station_id, [0])

        
        
        z_thresh = (abs(avg_z) + self.z_thresh) / defs.cvac
         
        z_thresh_up = (abs(avg_z) - self.z_thresh) / defs.cvac
        

        row, col = intmap["map"].shape
        cols = np.where(np.logical_and(intmap["z"].flatten() >= z_thresh, intmap["z"].flatten() <= z_thresh_up))        
        surf_array = (intmap["map"])[:row, min(cols[0]):max(cols[0])+1]
        max_surf_corr = np.max(surf_array)

        maxind = np.unravel_index(np.argmax(surf_array), surf_array.shape)
        max_z = intmap["z"].flatten()[min(cols[0]):max(cols[0])+1][maxind[1]]
        max_r = intmap["r"].flatten()[:row][maxind[0]]
    

        cols2 = np.where(intmap["z"].flatten() >= z_thresh)
        surf_array2 = (intmap["map"])[:row, min(cols2[0]):max(cols2[0])+1]
        max_surf_corr_2 = np.max(surf_array2)
        maxind2 = np.unravel_index(np.argmax(surf_array2), surf_array2.shape)
        max_z_2 = intmap["z"].flatten()[min(cols2[0]):max(cols2[0])+1][maxind2[1]],
        max_r_2 = intmap["r"].flatten()[maxind2[0]]
        
        

        if (maxcorr != 0):
            surf_corr_ratio = max_surf_corr / maxcorr 
            surf_corr_ratio_2 = max_surf_corr_2 / maxcorr
        else:
            surf_corr_ratio = np.inf 
            surf_corr_ratio_2 = np.inf
        
        
        return surf_corr_ratio, max_surf_corr, surf_corr_ratio_2, max_surf_corr_2, max_r * defs.cvac, max_z * defs.cvac, max_z_2[0] * defs.cvac, max_r_2 * defs.cvac

    


