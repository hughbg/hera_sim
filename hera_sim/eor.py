"""
A module containing functions for generating EoR-like signals.

Each model may take arbitrary parameters, but must return a 2D complex array containing the visibilities at the
requested baseline, for the requested lsts and frequencies.
"""

import numpy as np
from scipy import interpolate
import aipy
from scipy.signal import windows
from . import noise
from . import utils


def noiselike_eor(lsts, fqs, bl_vec, eor_amp=1e-5, min_delay=None, max_delay=None,
                  fringe_filter_type='tophat', **fringe_filter_kwargs):
    """
    Generate a noise-like, fringe-filtered EoR visibility.

    Args:
        lsts (ndarray): LSTs [radians]
        fqs (ndarray): frequencies [GHz]
        bl_vec (ndarray): East-North-Up (i.e. Topocentric) baseline vector in nanoseconds [East, North, Up]
        eor_amp (float): amplitude of EoR signal [arbitrary units]
        min_delay (float): minimum delay of signal to keep in nanosec (i.e. filter out below this delay)
        max_delay (float): maximum delay of signal to keep in nanosec (i.e. filter out above this delay)
        fringe_filter_type (str): type of fringe-rate filter, see utils.gen_fringe_filter()
        fringe_filter_kwargs: kwargs given fringe_filter_type, see utils.gen_fringe_filter()

    Returns: 
        vis (ndarray): simulated complex visibility

    Notes:
        Based on the order of operations (delay filter then fringe-rate filter),
        modes outside of min and max delay will contain some spillover power due
        to the frequency-dependent nature of the fringe-rate filter.
    """
    # generate white noise in frate and freq space
    data = noise.white_noise((len(lsts), len(fqs))) * eor_amp

    # filter across frequency
    delay_filter = utils.gen_delay_filter(fqs, 1e10, filter_type='tophat')
    dlys = np.fft.fftfreq(len(fqs), fqs[1] - fqs[0])
    if min_delay is not None:
        delay_filter[np.abs(dlys) < min_delay] = 0.0
    if max_delay is not None:
        delay_filter[np.abs(dlys) > max_delay] = 0.0
    data = np.fft.fft(np.fft.ifft(data, axis=1) * delay_filter, axis=1)

    # generate fringe filter in frate & freq space
    fringe_filter = utils.gen_fringe_filter(lsts, fqs, np.abs(bl_vec[0]), filter_type=fringe_filter_type, **fringe_filter_kwargs)
 
    # apply to data
    data = np.fft.fft(np.fft.ifft(data, axis=0) * fringe_filter, axis=0)

    return data
