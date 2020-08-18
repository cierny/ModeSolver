import numpy as np
from pathos.multiprocessing import _ProcessPool as Pool
from scipy.integrate import solve_ivp, quad, dblquad
from scipy.interpolate import splrep, splev
from scipy.optimize import least_squares, curve_fit
from scipy.special import kn

parallel = False
fiber = None
pool = None

def initialize(r_core, index, k, parallelize=False):
    global fiber
    fiber = (r_core, index, k)
    if parallelize:
        global parallel
        global pool
        parallel = True
        pool = Pool(initializer=initialize, initargs=fiber)

def iterated_task(task, stuff):
    return pool.map(task, stuff) if parallel else list(map(task, stuff))

def field(r, y, b, l):
    f, g = y
    r_core, index, k = fiber
    return [g, -g/r -f*((k*index(r))**2-(l/r)**2-b**2)]

def root_func(b, l, init):
    r_core, index, k = fiber
    r0, y0, dy0 = init
    w = np.sqrt(b**2 - (index(r_core)*k)**2)
    tester = solve_ivp(field, [r0, r_core], [y0, dy0], args=(b, l))
    corr = tester.y[0, -1] / kn(l, r_core*w)
    return tester.y[1, -1] + 0.5*corr*w*(kn(l-1, r_core*w)+kn(l+1, r_core*w))

def find_init(l, b_min):
    r0 = 1e-10
    y0 = 1
    dy0 = 0
    if l > 0:
        y0 = 1e-10
        dy0 = 1
        res = solve_ivp(field, [r0, fiber[0]], [y0, dy0], args=(b_min, l))
        c1 = np.abs(np.log10(np.abs(res.y[0, -1])))
        dy0 = np.maximum(1e-10, 10**(-c1))
        res = solve_ivp(field, [r0, fiber[0]], [y0, dy0], args=(b_min, l))
        c2 = np.abs(np.log10(np.abs(res.y[0, -1])))
        if c2 > 2:
            res = solve_ivp(field, [r0*10, fiber[0]], [y0, dy0], args=(b_min, l))
            c3 = np.abs(np.log10(np.abs(res.y[0, -1])))
            r0 = 10**(c2/(c2-c3) - 10)
    return r0, y0, dy0

def find_bzeros(l, init, b_min, b_max, m_max):
    npts = 5*m_max
    bzeros = []
    bs = np.linspace(b_max, b_min, npts)
    res = iterated_task(lambda b: root_func(b, l, init), bs)
    for idx in range(1, npts):
        if res[idx-1]*res[idx] < 0:
            bzeros.append(bs[idx-1])
    return bzeros

def solve_mode(bz, b_min, b_max, l, init):
    npts = 200
    r_core, index, k = fiber
    r0, y0, dy0 = init
    rz = np.arange(0, r0, r_core/npts)
    rs = np.linspace(r0, r_core, npts)
    beta = least_squares(root_func, bz, bounds=(b_min, b_max), args=(l, init)).x[0]
    core = solve_ivp(field, [r0, r_core], [y0, dy0], args=(beta, l), t_eval=rs)
    r_full = np.concatenate((rz, rs[1:]))
    y_full = np.concatenate((np.repeat(y0, len(rz)), core.y[0, 1:]))
    core_fit = splrep(r_full, y_full / np.max(y_full))
    w = np.sqrt(beta**2 - (k*index(r_core))**2)
    clad_fac = splev(r_core, core_fit) / kn(l, r_core*w)
    fit = lambda r: np.where(r < r_core, splev(r, core_fit), clad_fac*kn(l, r*w))
    return beta, fit

def find_modes():
    if fiber == None:
        raise Exception('Fiber not initialized')
    r_core, index, k = fiber
    m_max = 20
    b_max = k*index(0)
    b_min = k*index(r_core) + 1e-10
    modes = []
    l = 0
    while True:
        init = find_init(l, b_min)
        bzeros = find_bzeros(l, init, b_min, b_max, m_max)
        res = iterated_task(lambda b: solve_mode(b, b_min, b_max, l, init), bzeros)
        if len(res) > 0:
            b_max = res[0][0]
            m_max = len(res)
            for m, sol in enumerate(res):
                modes.append(((l,m+1), sol[0], sol[1]))
            l = l + 1
        else:
            return modes

def mfd(mode):
    s2x = quad(lambda r: (r*mode(r))**2, 0, np.inf)[0] / quad(lambda r: mode(r)**2, 0, np.inf)[0]
    return 4*np.sqrt(s2x)

def lp01(r_core, index, k):
    initialize(r_core, index, k)
    b_max = k*index(0)
    b_min = k*index(r_core) + 1e-10
    beta, fit = solve_mode(b_max, b_min, b_max, 0, find_init(0, b_min))
    rs = np.linspace(0, 2*r_core, 100)
    sigma_fit = curve_fit(lambda r,s: np.exp(-r**2/(2*s**2)), rs, fit(rs)**2)[0][0]
    return fit, mfd(fit), 4*sigma_fit

def coupling(psf_re, psf_im, modes, rs, r_max, bend=None, tol=1e-1):
    r_core, index, k = fiber
    bounds = [0, 2*np.pi, 0, r_max]
    if bend != None:
        e1 = index(r_core)**2
        e2 = index(0)**2
        modes = list(filter(lambda m: ((m[1]*bend/(k*(bend+r_core)))**2-e1)/(e2-e1) > 0, modes))
    results = np.zeros(len(rs))
    psf_int = lambda r: r*(psf_re(r)**2 + psf_im(r)**2)
    psf_res = 2*np.pi*quad(psf_int, 0, r_max, epsabs=tol)[0]
    def fib_task(mode):
        lm, beta, fit = mode
        return dblquad(lambda r,t: r*(fit(r)*np.cos(lm[0]*t))**2, *bounds, epsabs=tol)[0]
    fib_res = iterated_task(fib_task, modes)
    def cross_task(fit, l, mode_res, r0):
        cross_int_re = lambda r,t: r*fit(r)*np.cos(l*t)*psf_re(np.sqrt(r**2+r0**2-2*r*r0*np.cos(t)))
        cross_int_im = lambda r,t: r*fit(r)*np.cos(l*t)*psf_im(np.sqrt(r**2+r0**2-2*r*r0*np.cos(t)))
        cross_res_re = dblquad(cross_int_re, *bounds, epsabs=tol)[0]**2
        cross_res_im = dblquad(cross_int_im, *bounds, epsabs=tol)[0]**2
        ni = (cross_res_re + cross_res_im)/(psf_res*mode_res)
        return ni if l==0 else ni/2
    for idx, mode in enumerate(modes):
        lm, beta, fit = mode
        c_task = lambda r: cross_task(fit, lm[0], fib_res[idx], r)
        results = results + np.array(iterated_task(c_task, rs))
    return results
