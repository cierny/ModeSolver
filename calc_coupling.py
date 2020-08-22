import numpy as np
import modes as md
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
import time

r_core = 25
n_clad = 1.45
wl_tx = 0.78
wl_rx = 0.98

R = 30e3
w0 = 1.5e3
mag = 20
paa = 50e-6

# cut = 0.73
# NA = cut*2.405/(2*np.pi*r_core);
# n_core = np.sqrt(n_clad**2 + NA**2)
#
# def index(r):
#     if r < r_core:
#         return n_core
#     else:
#         return n_clad

def index(r):
    if r < r_core:
        return n_clad + 0.0145*(1 - (r/r_core)**2)
    else:
        return n_clad

def read_psf(filename):
    r_samp = []
    p_samp = []
    with open(filename,'r') as f_psf:
        reading = False
        start_point = 0
        while True:
            line = f_psf.readline()
            if not line:
                break
            if line.startswith('Image grid size:'):
                start_point = line.split(' ')[3]
            if int(start_point) > 0 and line.startswith(start_point):
                reading = True
            if reading:
                line = line.replace(' ','').split('	')
                r_samp.append(float(line[1]))
                p_samp.append(float(line[2]))
    return np.array(r_samp), np.array(p_samp)

if __name__ == '__main__':
    f_col = 33.5267
    dr_max = f_col*mag*paa*1e3
    r_im, p_im = read_psf('psf_im.txt')
    r_re, p_re = read_psf('psf_re.txt')
    psf_max = np.max(r_re)
    psf_im = InterpolatedUnivariateSpline(r_im, p_im, ext=1)
    psf_re = InterpolatedUnivariateSpline(r_re, p_re, ext=1)

    # print(n_core)
    tick = time.time()
    fit, mfd1, mfd2 = md.lp01(r_core, index, 2*np.pi/wl_tx)
    print('Tx w0 (s2x): %f' % (mfd1/2000))
    print('Tx w0 (fit): %f' % (mfd2/2000))
    print('f_col est: %f' % (w0*np.pi*mfd2/(2*wl_tx)*1e-3))

    print('Solving modes...')
    md.initialize(r_core, index, 2*np.pi/wl_rx, True)
    modes = md.find_modes(bend=R)
    print('Rx modes: %d' % len(modes))

    rs = np.linspace(0, dr_max, 50)
    res = md.coupling(psf_re, psf_im, psf_max, modes, rs, 10*r_core, atol=1)
    print("--- %s seconds ---" % (time.time() - tick))
    np.save('res_grin25', np.vstack((rs,res)))
