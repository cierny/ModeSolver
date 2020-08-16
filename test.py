import numpy as np
import modes as md

r_core = 25
n_clad = 1.45
wl = 1.55

def index(r):
    if r <= r_core:
        return n_clad + 0.0145*(1 - (r/r_core)**2)
    else:
        return n_clad

if __name__ == '__main__':
    md.initialize(r_core, index, 2*np.pi/wl, True)
    modes = md.find_modes()
    print(len(modes))
    # print(md.lp01_mfd(r_core, index, 2*np.pi/wl))
