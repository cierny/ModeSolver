import numpy as np
import modes as md
import time

r_core = 25
n_clad = 1.45
wl = 1.55
k = 2*np.pi/wl

def index(r):
    if r < r_core:
        return n_clad + 0.0145*(1 - (r/r_core)**2)
    else:
        return n_clad

def gauss(r, sigma):
    return np.exp(-r**2/(2*sigma**2))

if __name__ == '__main__':
    R = 30e3
    tick = time.time()
    md.initialize(r_core, index, k, True)
    modes = md.find_modes()
    print(len(modes))
    # print(md.lp01_mfd(r_core, index, k))
    print("--- %s seconds ---" % (time.time() - tick))
    tick = time.time()
    rs = np.linspace(0, 35, 100)
    psf = lambda r: gauss(r, 7.738/np.sqrt(2))
    res = md.coupling(psf, modes, rs, 2*r_core)
    print("--- %s seconds ---" % (time.time() - tick))
    np.save('res2', res)
