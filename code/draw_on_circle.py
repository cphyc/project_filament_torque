import numpy as np
from scipy.special import spherical_jn as j
from tqdm import tqdm
import os

import angular_utils as tu
from utils import W1TH, W1G, trapz, get_maxis

# Radius of circle
Rc = 10  # radius of circle, in Mpc/h
R  = 5   # smoothing scale, in Mpc/h
k, Pk, _ = np.loadtxt('power.dat', skiprows=1).T.astype(np.float32)
Pk *= 2*np.pi**2 * 4*np.pi


###############################################################################
# Compute the covariance matrix
###############################################################################
# Draw positions on the circle
phi = np.linspace(0, 2*np.pi, 200, endpoint=False, dtype=np.float32)

# Compute 2-point distances
dist = Rc * (1 - np.cos(phi[:, None] - phi[None, :]))

# Compute the correlation
print("Computing correlation coefficients")
kr = k[None, None, :] * dist[:, :, None]
kR = k[None, None, :]*R
k2Pk = k[None, None, :]**2*Pk[None, None, :]
j1kr = np.where(kr == 0, 1/3, j(1, kr) / kr)
W1 = W1G
xi00 = trapz(k2Pk * W1(kR)**2 * j(0, kr) / (2*np.pi**2), k)
# xi11 = trapz(k2Pk * W1(kR)**2 * j1kr / (2*np.pi**2), k)
# xi20 = trapz(k2Pk * W1(kR)**2 * j(2, kr) / (2*np.pi**2), k)

# Covariance is just xi00
cov = xi00
mean = np.zeros_like(phi)


###############################################################################
# Draw random sample
###############################################################################
phi1_f = np.empty(0)
phi2_f = np.empty(0)
v0_f, v1_f, v2_f = [np.empty(0) for _ in range(3)]
Ntot = int(1e6)
Nperdraw = Ntot
prog = tqdm(total=Ntot)
u, s, v = np.dual.svd(cov)


def multivariate_normal(size):
    shape = [size, len(mean)]
    x = np.random.standard_normal(size=shape)
    x = np.dot(x, np.sqrt(s)[:, None] * v)
    x += mean
    return x


np.random.seed(16091992)

try:
    if os.path.exists('data.h5'):
        phi1_f, phi2_f, v0_f, v1_f, v2_f = tu.load_data()
        prog.update(len(phi1_f))

    while len(phi1_f) < Ntot:
        sample = multivariate_normal(Nperdraw).astype(np.float32)
        # sample = np.random.multivariate_normal(mean, cov, size=Nperdraw)
        m0, m1, m2, v0, v1, v2 = get_maxis(sample)
        mask = m0 >= 0
        phi0 = phi[m0[mask]]
        phi1 = phi[m1[mask]]
        phi2 = phi[m2[mask]]
        v0 = v0[mask]
        v1 = v1[mask]
        v2 = v2[mask]

        phi1 -= phi0
        phi2 -= phi0

        phi1 = np.mod(phi1, 2*np.pi)
        phi2 = np.mod(phi2, 2*np.pi)

        phi2 = np.where(phi1 > np.pi, 2*np.pi-phi2, phi2)
        phi1 = np.where(phi1 > np.pi, 2*np.pi-phi1, phi1)

        phi1_f = np.concatenate([phi1_f, phi1])
        phi2_f = np.concatenate([phi2_f, phi2])
        v0_f = np.concatenate([v0_f, v0])
        v1_f = np.concatenate([v1_f, v1])
        v2_f = np.concatenate([v2_f, v2])
        prog.update(len(phi1))

finally:
    tu.save_data(phi1_f, phi2_f, v0_f, v1_f, v2_f)
    tu.plot_histogram(phi1_f, phi2_f, v0_f, v1_f, v2_f)
    tu.plot_surface(phi1_f, phi2_f, v0_f, v1_f, v2_f)
