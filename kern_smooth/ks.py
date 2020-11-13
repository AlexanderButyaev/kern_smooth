'''
Functions for Kernel Smoothing and density estimation.
Transformed R and Fortran functions into Python(3) code.

@ref Wand, M.P. and Jones, M.C. (1995) "Kernel Smoothing".
@ref R::grDevices package
@ref R::KernSmooth package
@author Alexander Butyaev
'''
import math
import numbers
import numpy as np
from pandas import cut
from scipy.stats.mstats import mquantiles
from scipy.stats import norm


def densCols(x, y=None, nbin=128, bandwidth=None, return_dens_matrix=False):
    '''
    Produces a vector containing numbers which encode the local densities at each point in a scatterplot.

    x, y : 1D numpy array with coordinates of the points density will be estimated on
    nbin : [optional] int or [int, int] - number of bins along each axis
        (in case of single value - [nbin, nbin] will be used). Default value 128.
    bandwidth : [optional] numeric vector (len of 1 or 2) of smoothing bandwidth.

    Returns: numpy array with numerical representation (in range [0,1]) of point densities.
    Attention: For return value numpy.nan values are allowed in case of nan / infinite values in original dataset 
    Source: R::grDevices::densCols

    Added functionality: if return_dens_matrix == True returs tuple:
        1) [as described above]
        2) density matrix of size nbin (squared)
    '''
    x = x
    y = y if not y is None else x
    # deal with NA, etc
    select = np.isfinite(x) & np.isfinite(y)
    inds = [i for i, s in enumerate(select) if s]
    x = np.dstack((x, y))[0][inds, :]
    # create density map
    axes, fhat, bandwidth = smoothScatterCalcDensity(x, nbin, bandwidth)
    # bin  x- and y- values
    xbin = cut(x[:, 0], __mk_breaks(axes[0])).codes
    ybin = cut(x[:, 1], __mk_breaks(axes[1])).codes
    dens = fhat[xbin, ybin]
    dens[np.isnan(dens)] = 0
    # transform densities to colors ()
    colpal = cut(dens, len(dens)).codes
    cols = np.empty(select.shape[0])
    cols.fill(np.nan)
    cols[select] = colpal / (len(dens) - 1.) if len(dens) > 1 else colpal
    if return_dens_matrix:
        return cols, fhat
    return cols


def __mk_breaks(u):
    '''
    Produces grid break points for further binning

    u : numpy ndarray type
    '''
    return u - (np.max(u) - np.min(u)) / (u.size - 1) / 2


def linbin2D(X, Y, n_break_poins, rangex=[[0, 1], [0, 1]]):
    '''
    Applies linear binning strategy to a bivariate data set.

    Returns: 'n_break_poins'' shaped matrix with data binning
    py adoptation of fortran's KernSmooth::linbin2D
    '''
    M0, M1 = int(n_break_poins[0]), int(n_break_poins[1])
    n = len(X)
    if len(Y) != len(X):
        raise ValueError("X and Y should have the same length")
    gcnts = np.zeros(M0 * M1)
    # gcnts = np.zeros( M0 * M1 + 1)
    # find a grid step
    delta1 = (rangex[0][1] - rangex[0][0]) / (M0 - 1)
    delta2 = (rangex[1][1] - rangex[1][0]) / (M1 - 1)
    # define bin indices along both axis
    _X = (X - rangex[0][0]) / delta1
    _Y = (Y - rangex[1][0]) / delta2
    # truncate points which indices are out of the specified range 0<= int(_x)
    # <M0 & 0<= int(_y) <M1
    _X_int = np.trunc(_X)
    _Y_int = np.trunc(_Y)
    _X = _X[np.logical_and(np.logical_and(
        _X_int >= 0, _X_int < M0), np.logical_and(_Y_int >= 0, _Y_int < M1))]
    _Y = _Y[np.logical_and(np.logical_and(
        _X_int >= 0, _X_int < M0), np.logical_and(_Y_int >= 0, _Y_int < M1))]
    _X_dec, _X_int = np.modf(_X)
    _Y_dec, _Y_int = np.modf(_Y)
    _X_int = _X_int.astype(int)
    _Y_int = _Y_int.astype(int)
    # find indices for linear binning (4)
    inds1 = M0 * (_Y_int) + _X_int
    inds2 = M0 * (_Y_int) + _X_int + 1
    inds3 = M0 * (_Y_int + 1) + _X_int
    inds4 = M0 * (_Y_int + 1) + _X_int + 1
    for li1, li2, rem1, rem2, ind1, ind2, ind3, ind4 in zip(
            _X_int, _Y_int, _X_dec, _Y_dec, inds1, inds2, inds3, inds4):
        gcnts[ind1] = gcnts[ind1] + (1 - rem1) * (1 - rem2)
        gcnts[ind2] = gcnts[ind2] + rem1 * (1 - rem2)
        gcnts[ind3] = gcnts[ind3] + (1 - rem1) * rem2
        gcnts[ind4] = gcnts[ind4] + rem1 * rem2
    return gcnts.reshape((M0, M1))


def smoothScatterCalcDensity(x, nbin, bandwidth=None, rangex=None):
    '''
    Preprocessing step for kde function:
        'nbin' initialization,
        'bandwidth' initializtion and validation

    x : numpy array [shape = (2,N)] - array with coordinates of the points
    nbin : int or [int, int] - number of bins along both axis (in case single value - [nbin, nbin] is used)
    bandwidth : [optional] numeric positive array of size 2 with smoothing bandwidth

    return  axes - pair of lists with axis breakpoints
            fhat - binning Kernel Density Estimation matrix (squared)
            bandwidth - smoothing bandwidth (own estimation in case of initial bandwidth = None)
    Source: R::KernSmooth::smoothScatterCalcDensity
    '''
    if isinstance(nbin, numbers.Number):
        nbin = (nbin, nbin)
    elif (isinstance(nbin, list) and len(nbin) == 1) or (isinstance(nbin, np.ndarray) and len(nbin) == 1):
        nbin = (nbin[0], nbin[0])
    if len(nbin) != 2 or not(isinstance(nbin[0], numbers.Number) and isinstance(nbin[1], numbers.Number)):
        raise ValueError("'nbin' must be numeric of length 1 or 2")
    if bandwidth is None:
        # R compatibility
        q_data = mquantiles(x, prob=[0.05, 0.95],
                            alphap=1, betap=1, axis=0).data
        bandwidth = np.diff(q_data, axis=0) / 25
        bandwidth[bandwidth == 0] = 1
        bandwidth = bandwidth[0]
    else:
        if not (isinstance(bandwidth, numbers.Number) or isinstance(bandwidth, np.ndarray)):
            raise ValueError("'bandwidth' must be numeric")
        if isinstance(bandwidth, np.ndarray) and len(bandwidth[bandwidth <= 0]) > 0:
            raise ValueError("'bandwidth' must be positive")
    rv = bkde2D(x, bandwidth=bandwidth, gridsize=nbin, rangex=rangex)
    # return axes, fhat, bandwidth
    return rv[0], rv[1], bandwidth


def bkde2D(x, bandwidth, gridsize=(51, 51), rangex=None):
    '''
    Produces binning Kernel Density Estimation

    x - (2,N) shaped numpy array containing the observations
    bandwidth - numeric vector of length 2, containing the bandwidth to be used along each axis
    gridsize - pair of number of equally spaced points along both axis
    rangex - a list containing two vectors, where each vector contains the minimum and maximum values
        of x at which to compute the estimate for each direction.
        The default minimum in each direction is minimum data value minus 1.5 times the bandwidth for that direction.
        The default maximum is the maximum data value plus 1.5 times the bandwidth for that direction.

    Returns: axes - pair of lists with axis breakpoints
             rp - binning Kernel Density Estimation matrix (squared)
    Attention: data with x values outside the range specified by rangex are ignored.
    '''
    n, dims = x.shape  # @TODO: choose beter name (Compatibility)
    M = np.array(gridsize)
    h = bandwidth
    tau = 3.4  # For bivariate normal kernel.

    # Use same bandwidth in each direction
    # if only a single bandwidth is given.
    if isinstance(h, numbers.Number):
        if h <= 0:
            raise ValueError("'bandwidth' must be strictly positive")
        h = (h, h)
    elif (isinstance(h, list) and len(h) == 1) or (isinstance(h, np.ndarray) and len(h) == 1):
        h = (h[0], h[0])
    mins = np.min(x, axis=0)
    maxs = np.max(x, axis=0)
    if not rangex:
        rangex = [0, 0]
        for _id in range(2):
            rangex[_id] = [mins[_id] - 1.5 * h[_id], maxs[_id] + 1.5 * h[_id]]
    a = (rangex[0][0], rangex[1][0])
    b = (rangex[0][1], rangex[1][1])
    # Set up grid points and bin the data
    gpoints1 = np.linspace(a[0], b[0], M[0])
    gpoints2 = np.linspace(a[1], b[1], M[1])
    # Linear binning strategy
    gcounts = linbin2D(x[:, 0], x[:, 1], n_break_poins=M,
                       rangex=[[a[0], b[0]], [a[1], b[1]]])
    # Compute kernel weights
    gcounts = gcounts.T
    L = np.zeros(2, dtype=np.int)
    kapid = [0, 0]
    for _id in range(2):
        L[_id] = min(math.floor(tau * h[_id] * (M[_id] - 1) /
                                (b[_id] - a[_id])), M[_id] - 1)
        lvecid = np.array(range(0, L[_id] + 1))
        facid = (b[_id] - a[_id]) / (h[_id] * (M[_id] - 1))
        z = norm.pdf(lvecid * facid) / h[_id]  # supposed to be matrix
        tot = (sum(z) + sum(z[1:][::-1])) * facid * h[_id]
        kapid[_id] = z / tot
    kapp = np.outer(kapid[0], kapid[1]) / n
    if min(L) == 0:
        raise ValueError(
            "Binning grid too coarse for current (small) bandwidth: consider increasing 'gridsize'")
    # Now combine weight and counts using the FFT to obtain estimate
    P = 2**(np.ceil(np.log2(M + L)))   # smallest powers of 2 >= M+L
    L1, L2 = L[0], L[1]
    M1, M2 = M[0], M[1]
    P1, P2 = P[0], P[1]
    if not (P1.is_integer() and P2.is_integer()):
        raise ValueError("something is wrong. P1, P2 should be integers!")
    P1, P2 = int(P1), int(P2)
    rp = np.zeros((P1, P2))
    rp[0: (L1 + 1), 0: (L2 + 1)] = kapp
    if L1:
        rp[(P1 - L1):P1, 0:(L2 + 1)] = kapp[L1:0:-1, 0:(L2 + 1)]
    if L2:
        rp[:, (P2 - L2):P2] = rp[:, (L2):0:-1]
    # wrap-around version of "kapp"
    sp = np.zeros((P1, P2))
    sp[0:M1, 0:M2] = gcounts
    # zero-padded version of "gcounts"
    # Obtain FFT's of r and s
    rp = np.fft.fft2(rp)
    sp = np.fft.fft2(sp)
    # invert element-wise product of FFT's
    # no normalization is required - ifft2's post processing step
    rp = np.fft.ifft2(rp * sp).real[0:M1, 0:M2]
    # Ensure that rp is non-negative
    rp[rp <= 0] = 0
    axes = (gpoints1, gpoints2)
    return axes, rp
