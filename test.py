import numpy as np
import modes as md
from scipy.interpolate import InterpolatedUnivariateSpline
import time

r_core = 2.2
n_clad = 1.45
wl_tx = 0.78
wl_rx = 0.98

R = 30e3
w0 = 1.5e3
mag = 20
paa = 50e-6

cut = 0.73
NA = cut*2.405/(2*np.pi*r_core);
# NA = 0.2

n_core = np.sqrt(n_clad**2 + NA**2)

def index(r):
    if r < r_core:
        return n_core
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
    r_im, p_im = read_psf('fft_im.txt')
    r_re, p_re = read_psf('fft_re.txt')
    psf_im = InterpolatedUnivariateSpline(r_im, p_im, ext=1)
    psf_re = InterpolatedUnivariateSpline(r_re, p_re, ext=1)

    md.initialize(r_core, index, 2*np.pi/wl_rx, True)
    modes = md.find_modes()
    print('Modes: %d' % len(modes))

    fit, mfd1, mfd2 = md.lp01(r_core, index, 2*np.pi/wl_rx)
    print('MFD (s2x): %f' % mfd1)
    print('MFD (fit): %f' % mfd2)
    print('NA (s2x): %f' % np.sin((2*wl_rx)/(np.pi*mfd1)))
    print('NA (fit): %f' % np.sin((2*wl_rx)/(np.pi*mfd2)))

    tick = time.time()
    # rs = np.linspace(0, 10, 100)
    res = md.coupling(psf_re, psf_im, modes, [0], np.inf)
    print('Coupling: %f' % res[0])
    # print("--- %s seconds ---" % (time.time() - tick))
    # np.save('res', res)
