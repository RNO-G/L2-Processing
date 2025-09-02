import numpy as np

cvac = 0.3

def ior_exp1(z):

    # Note: z is given in natural feet, convert to meter
    def iorfunc(z):
        A = 1.78
        B = 1.326
        C = 0.0202
        return A - (A - B) * np.exp(C * z * cvac)

    iorvals = iorfunc(z)
    iorvals[z > 0] = 1.0
    return iorvals


def ior_greenland_simple(z):
    def iorfunc(z):
        #return 1.78
        
        A = 602
        rho0 = 917
        #rho0 = (0.78 * 602)/0.51
        z0 = 37.25
        rho = rho0 - A * np.exp(z * cvac / z0)
        return 1 + 0.78 * (rho / rho0)
        
    
    iorvals = iorfunc(z)
    iorvals[z > 0] = 1.0
    return iorvals 


def ior_exp3(z):

    # Note: z is given in natural feet, convert to meter
    def iorfunc_snow(z):
        return 1.52737 - 0.298415 * np.exp(0.107158 * z * cvac)

    def iorfunc_firn(z):
        return 1.89275 - 0.521529 * np.exp(0.0136059 * z * cvac)

    def iorfunc_bubbly(z):
        return 1.77943 - 1.576 * np.exp(0.0403732 * z * cvac)

    z1 = -14.9 / cvac   # transition between snow and firn
    z2 = -80.5 / cvac   # transition between firn and bubbly ice

    snow_mask = np.argwhere(np.logical_and(z <= 0, z > z1))
    firn_mask = np.argwhere(np.logical_and(z <= z1, z > z2))
    bubbly_mask = np.argwhere(z <= z2)

    iorvals = np.zeros_like(z)
    iorvals[snow_mask] = iorfunc_snow(z[snow_mask])
    iorvals[firn_mask] = iorfunc_firn(z[firn_mask])
    iorvals[bubbly_mask] = iorfunc_bubbly(z[bubbly_mask])
    iorvals[z > 0] = 1.0

    return iorvals

"""
z_vals = np.array([-8000.        ,-7724.25747525,-7448.5149505 ,-7172.50575019,
 -6896.76322544,-6620.75402513,-6345.01150038,-6069.00230008,
 -5793.25977533,-5517.25057502,-5241.50805027,-4965.76552552,
 -4689.75632521,-4414.01380046,-4138.00460015,-3862.2620754 ,
 -3586.2528751 ,-3310.51035035,-3034.50115004,-2758.75862529,
 -2483.01610054,-2207.00690023,-1931.26437548,-1655.25517517,
 -1379.51265042,-1103.50345012, -827.76092536, -551.75172506,
  -276.00920031])

iorvs = ior_exp3(z_vals * 0.3)

for i in range(len(iorvs)):
    print(iorvs[i], z_vals[i] * 0.3)
"""
