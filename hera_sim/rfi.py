"""A module for generating realistic HERA RFI."""

import attr
import numpy as np
from astropy.units import sday
from attr import validators as vld
from scipy import stats


@attr.s(frozen=True, kw_only=True)
class RfiStation:
    """
    Class for representing an RFI transmitter.

    Args:
        fq0 (float): GHz
            center frequency of the RFI transmitter
        duty_cycle (float): default=1.
            fraction of times that RFI transmitter is on
        strength (float): Jy, default=100
            amplitude of RFI transmitter
        std (float): default=10.
            standard deviation of transmission amplitude
        timescale (scalar): seconds, default=100.
            timescale for which signal is typically coherently "on"
    """
    fq0 = attr.ib(converter=float, validator=vld.instance_of(float))
    width = attr.ib(1e6, converter=float)
    duty_cycle = attr.ib(1.0, converter=float)
    strength = attr.ib(100.0, converter=float)
    std = attr.ib(10.0, converter=float)
    timescale = attr.ib(100.0, converter=float)
    phase_dstn = attr.ib(stats.uniform(-np.pi, np.pi),
                         validator=vld.instance_of(stats.rv_frozen))

    @width.validator
    def _wdth_validator(self, att, val):
        assert 0 < val < self.fq0

    @duty_cycle.validator
    def _duty_cycle_validator(self, att, val):
        assert 0 <= val <= 1

    @strength.validator
    def _strength_validator(self, att, val):
        assert val >= 0

    @std.validator
    def _std_validator(self, att, val):
        assert val > 0
        assert val < self.strength / 5

    @timescale.validator
    def _timescale_validator(self, att, val):
        assert val > 0

    def get_probability_of_being_on(self, integration_time):
        """
        The probability of being "on" is not exactly the
        duty cycle, because once on, they're typically on
        for a while.

        The probability is closer to X/(X - L*X + L), where
        X is the duty cycle and L is the typical length (in bins)
        """
        l = int(self.timescale / integration_time)

        if l < 1:
            return self.duty_cycle
        else:
            return self.duty_cycle / (self.duty_cycle - l * self.timescale + l)

    def determine_channels(self, freqs, nsigma=5):
        return np.argwhere(
            np.logical_and(
                self.fq0 - self.width * nsigma <= freqs,
                freqs <= self.fq0 + self.width * nsigma)
        )[:, 0]

    def lsts_to_time_offsets(self, lsts):

        sid_day = sday.to("s")
        dlst_to_sec = sid_day / (2 * np.pi)
        times = lsts * dlst_to_sec
        times -= times[0]

        wraps = np.argwhere(np.diff(times) < 0)[:, 0]
        for w in wraps:
            times[(1 + w):] += sid_day

        return times

    def determine_times(self, lsts):
        self.validate_lsts(lsts)

        # get a series of time offsets (from first lsts)
        times = self.lsts_to_time_offsets(lsts)

        intg_time = times[1] - times[0]
        prob = self.get_probability_of_being_on(intg_time)

        # Determine which times should be "on"
        current_indx = 0

        on_lsts = []

        while current_indx < len(times) - 1:
            on = np.random.uniform() < prob

            if on:
                max_indx = np.argwhere(times > times[current_indx] + self.timescale)[0, 0] - 1
                on_lsts += range(current_indx, max_indx)
                current_indx = max_indx + 1
            else:
                current_indx += 1

        on_lsts = np.array(on_lsts)
        return on_lsts[on_lsts < len(lsts)]

    def validate_lsts(self, lsts):
        dlst = np.diff(lsts)
        assert np.all(np.logical_or(dlst > 0, dlst > 2 * np.pi)), "LSTs must be in increasing order"

        ddlst = np.diff(dlst)
        assert np.allclose(ddlst, 0), "LSTs must be regularly spaced"

    def gen_rfi(self, freqs, lsts, rfi=None):
        """
        Generate an (NTIMES,NFREQS) waterfall containing RFI.

        Args:
            lsts (array-like): shape=(NTIMES,), radians
                local sidereal times of the waterfall to be generated.
            freqs (array-like): shape=(NFREQS,), GHz
                the spectral frequencies of the waterfall to be generated.
            rfi (array-like): shape=(NTIMES,NFREQS), default=None
                an array to which the RFI will be added.  If None, a new array
                is generated.
        Returns:
            rfi (array-like): shape=(NTIMES,NFREQS)
                a waterfall containing RFI
        """
        # Initialize RFI array
        if rfi is None:
            rfi = np.zeros((lsts.size, freqs.size), dtype=np.complex)
        assert rfi.shape == (lsts.size, freqs.size), "rfi is not shape (lsts.size, fqs.size)"

        channels = self.determine_channels(freqs)
        if len(channels) == 0:
            # RFI does not appear in the set of frequencies given
            return rfi

        times = self.determine_times(lsts)

        # Draw amplitudes and phases
        phases = self.phase_dstn.rvs(size=(times.size, channels.size))
        amplitudes = stats.norm(loc=self.strength, scale=self.std).rvs(size=(times.size, channels.size))

        rfi[times][channels] += amplitudes * np.exp(2j * np.pi * phases)
        return rfi


HERA_RFI_STATIONS = [
    RfiStation(
        fq0=1.060e-01, width=7.581e-05, duty_cycle=1.00, strength=1.999e+05,
        std=1.097e+04, timescale=np.inf, phase_dstn=stats.norm(0.51, 0.04)),
    RfiStation(
        fq0=1.376e-01, width=1.171e-04, duty_cycle=0.74, strength=4.243e+04,
        std=2.714e+04, timescale=6.72e+02, phase_dstn=stats.uniform(-np.pi, np.pi)),
    RfiStation(
        fq0=1.372e-01, width=4.636e-05, duty_cycle=0.40, strength=4.419e+04,
        std=3.409e+04, timescale=3.62e+02, phase_dstn=stats.uniform(-np.pi, np.pi)),
    RfiStation(
        fq0=1.373e-01, width=7.222e-05, duty_cycle=0.30, strength=7.749e+03,
        std=1.151e+04, timescale=2.30e+02, phase_dstn=stats.uniform(-np.pi, np.pi)),
    RfiStation(
        fq0=1.371e-01, width=4.603e-05, duty_cycle=0.53, strength=3.031e+02,
        std=4.155e+02, timescale=3.06e+02, phase_dstn=stats.uniform(-np.pi, np.pi)),
    RfiStation(
        fq0=1.370e-01, width=5.304e-05, duty_cycle=0.48, strength=1.731e+02,
        std=1.714e+02, timescale=4.58e+02, phase_dstn=stats.norm(0.69, 0.83)),
    RfiStation(
        fq0=1.499e-01, width=3.489e-05, duty_cycle=1.00, strength=1.258e+03,
        std=4.844e+02, timescale=np.inf, phase_dstn=stats.norm(-2.41, 0.30)),
    RfiStation(
        fq0=1.459e-01, width=5.471e-05, duty_cycle=0.30, strength=2.111e+02,
        std=2.644e+02, timescale=1.39e+02, phase_dstn=stats.uniform(-np.pi, np.pi)),
    RfiStation(
        fq0=1.832e-01, width=5.197e-05, duty_cycle=1.00, strength=7.962e+02,
        std=9.578e+02, timescale=np.inf, phase_dstn=stats.norm(0.22, 0.21)),
    RfiStation(
        fq0=1.458e-01, width=5.001e-05, duty_cycle=0.32, strength=2.501e+02,
        std=3.275e+02, timescale=1.51e+02, phase_dstn=stats.uniform(-np.pi, np.pi)),
    RfiStation(
        fq0=1.050e-01, width=7.348e-05, duty_cycle=1.00, strength=1.075e+03,
        std=1.044e+03, timescale=np.inf, phase_dstn=stats.norm(-1.62, 0.05)),
    RfiStation(
        fq0=1.457e-01, width=4.038e-05, duty_cycle=0.27, strength=1.087e+02,
        std=1.243e+02, timescale=7.13e+01, phase_dstn=stats.uniform(-np.pi, np.pi)),
    RfiStation(
        fq0=1.912e-01, width=1.617e-04, duty_cycle=1.00, strength=3.563e+02,
        std=3.740e+02, timescale=np.inf, phase_dstn=stats.norm(1.05, 0.30)),
    RfiStation(
        fq0=1.055e-01, width=6.966e-05, duty_cycle=0.99, strength=1.179e+02,
        std=1.335e+02, timescale=1.20e+03, phase_dstn=stats.norm(1.40, 0.10)),
    RfiStation(
        fq0=1.064e-01, width=7.175e-05, duty_cycle=0.71, strength=1.400e+02,
        std=1.545e+02, timescale=9.24e+01, phase_dstn=stats.norm(-1.45, 0.16)),
    RfiStation(
        fq0=1.751e-01, width=5.691e-05, duty_cycle=1.00, strength=1.016e+02,
        std=9.531e+01, timescale=np.inf, phase_dstn=stats.norm(-1.57, 0.80)),
    RfiStation(
        fq0=1.063e-01, width=6.893e-05, duty_cycle=0.92, strength=9.980e+01,
        std=1.057e+02, timescale=3.43e+02, phase_dstn=stats.norm(2.44, 0.14)),
    RfiStation(
        fq0=1.069e-01, width=1.283e-04, duty_cycle=0.86, strength=9.021e+01,
        std=8.039e+01, timescale=1.78e+02, phase_dstn=stats.norm(-2.77, 0.26)),
    RfiStation(
        fq0=1.052e-01, width=7.576e-05, duty_cycle=1.00, strength=4.233e+01,
        std=5.071e+01, timescale=8.50e+03, phase_dstn=stats.norm(-2.59, 0.30)),
    RfiStation(
        fq0=1.072e-01, width=8.233e-05, duty_cycle=0.50, strength=3.351e+01,
        std=2.639e+01, timescale=5.99e+01, phase_dstn=stats.uniform(-np.pi, np.pi)),
    RfiStation(
        fq0=1.892e-01, width=4.631e-05, duty_cycle=1.00, strength=7.677e+01,
        std=7.196e+01, timescale=np.inf, phase_dstn=stats.norm(2.07, 0.21)),
    RfiStation(
        fq0=1.057e-01, width=4.545e-05, duty_cycle=0.92, strength=9.260e+00,
        std=6.854e+00, timescale=2.95e+02, phase_dstn=stats.uniform(-np.pi, np.pi)),
    RfiStation(
        fq0=1.057e-01, width=7.239e-05, duty_cycle=0.89, strength=9.449e+00,
        std=6.684e+00, timescale=2.85e+02, phase_dstn=stats.uniform(-np.pi, np.pi))
]


# XXX reverse lsts and fqs?
def rfi_stations(fqs, lsts, stations=HERA_RFI_STATIONS, rfi=None):
    """
    Generate an (NTIMES,NFREQS) waterfall containing RFI stations that
    are localized in frequency.

    Args:
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the waterfall to be generated.
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the waterfall to be generated.
        stations (iterable): list of 5-tuples, default=HERA_RFI_STATIONS
            a list of (FREQ, DUTY_CYCLE, STRENGTH, STD, TIMESCALE) tuples
            for RfiStations that will be injected into waterfall. Instead
            of a tuple, an instance of :class:`RfiStation` may be given.
        rfi (array-like): shape=(NTIMES,NFREQS), default=None
            an array to which the RFI will be added.  If None, a new array
            is generated.
    Returns:
        rfi (array-like): shape=(NTIMES,NFREQS)
            a waterfall containing RFI
    """
    for s in stations:
        if not isinstance(s, RfiStation):
            if len(s) != 5:
                raise ValueError("Each station must be a 5-tuple")

            s = RfiStation(*s)
        rfi = s.gen_rfi(fqs, lsts, rfi=rfi)
    return rfi


# XXX reverse lsts and fqs?
def rfi_impulse(fqs, lsts, rfi=None, chance=0.001, strength_mean=20.0,
                strength_std=0.6, integration_time=10.7,
                presence_frac=0.15, time_width_mean=10.7, time_width_std=0.6):
    """
    Generate an (NTIMES,NFREQS) waterfall containing RFI impulses that
    are localized in time but span the entire frequency band.

    Args:
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the waterfall to be generated.
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the waterfall to be generated.
        rfi (array-like): shape=(NTIMES,NFREQS), default=None
            an array to which the RFI will be added.  If None, a new array
            is generated.
        chance (float):
            the probability that a time bin will be assigned an RFI impulse
        strength_mean (float): Jy
            the mean strength of the impulse generated in each time/freq bin
        strength_std (float):
            the std of the strengths of the impulses, *in log space*.
        integration_time (float):
            the integration time of the observation (or width of the time bins) [sec]
        presence_frac (float):
            defines the fraction of a time-bin that a nominal RFI must be "on" for
            to render that time-bin as significantly RFI-contaminated.
        time_width_mean (float):
            the mean of the distribution of time widths of the impulses.
        time_width_std (float):
            the standard deviation of the distribution of time widths of the impulses
            *in log space*.

    Returns:
        rfi (array-like): shape=(NTIMES,NFREQS)
            a waterfall containing RFI

    See Also:
        rfi_blip: generate waterfalls with short-time impulses that span a wide but
                  not full selection of frequencies.
    """
    if rfi is None:
        rfi = np.zeros((lsts.size, fqs.size), dtype=np.complex)
    assert rfi.shape == (lsts.size, fqs.size), "rfi is not shape (lsts.size, fqs.size)"

    # Find a fiducial random list of times at which full-band impulses arrive
    impulse_times = np.where(np.random.uniform(size=lsts.size) <= chance)[0]

    if impulse_times.size == 0:
        return rfi

    # Now generate random time-bin-widths of these times.
    times = np.exp(np.random.normal(np.log(time_width_mean), time_width_std, size=len(impulse_times)))
    full_bin_times = times // integration_time
    full_bin_times[times % integration_time / integration_time > presence_frac] += 1

    for impulse_time, width in zip(impulse_times, full_bin_times):
        dlys = np.random.uniform(-300, 300, size=width)  # ns
        strength = np.exp(np.random.normal(np.log(strength_mean), scale=strength_std))
        impulses = strength * np.array([np.exp(2j * np.pi * dly * fqs) for dly in dlys])
        rfi[impulse_time:impulse_time + width] += impulses

    return rfi


# XXX reverse lsts and fqs?
def rfi_scatter(fqs, lsts, rfi=None, chance=0.0001, strength=10, std=10):
    """
    Generate an (NTIMES,NFREQS) waterfall containing RFI impulses that
    are localized in time but span the frequency band.

    Args:
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the waterfall to be generated.
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the waterfall to be generated.
        rfi (array-like): shape=(NTIMES,NFREQS), default=None
            an array to which the RFI will be added.  If None, a new array
            is generated.
        chance (float): default=0.0001
            the probability that a time/freq bin will be assigned an RFI impulse
        strength (float): Jy, default=10
            the average amplitude of the spike generated in each time/freq bin
        std (float): Jy, default = 10
            the standard deviation of the amplitudes drawn for each time/freq bin
    Returns:
        rfi (array-like): shape=(NTIMES,NFREQS)
            a waterfall containing RFI
    """
    if rfi is None:
        rfi = np.zeros((lsts.size, fqs.size), dtype=np.complex)
    assert rfi.shape == (lsts.size, fqs.size), "rfi shape is not (lsts.size, fqs.size)"

    rfis = np.where(np.random.uniform(size=rfi.size) <= chance)[0]
    rfi.flat[rfis] += np.random.normal(strength, std) * np.exp(
        2 * np.pi * 1j * np.random.uniform(size=rfis.size)
    )
    return rfi


def rfi_dtv(fqs, lsts, rfi=None, freq_min=.174, freq_max=.214, width=0.008,
            chance=0.0001, strength=10, strength_std=10):
    """
    Generate an (NTIMES, NFREQS) waterfall containing Digital TV RFI.

    DTV RFI is expected to be of uniform bandwidth (eg. 8MHz), in contiguous
    bands, in a nominal frequency range. Furthermore, it is expected to be
    short-duration, and so is implemented as randomly affecting discrete LSTS.

    There may be evidence that close times are correlated in having DTV RFI,
    and this is *not currently implemented*.

    Args:
        fqs (array-like): shape=(NFREQS,), GHz
            the spectral frequencies of the waterfall to be generated.
        lsts (array-like): shape=(NTIMES,), radians
            local sidereal times of the waterfall to be generated.
        rfi (array-like): shape=(NTIMES,NFREQS), default=None
            an array to which the RFI will be added.  If None, a new array
            is generated.
        freq_min, freq_max (float):
            the min and max frequencies of the full DTV band [GHz]
        width (float):
            Width of individual DTV bands [GHz]
        chance (float): default=0.0001
            the probability that a time/freq bin will be assigned an RFI impulse
        strength (float): Jy, default=10
            the average amplitude of the spike generated in each time/freq bin
        strength_std (float): Jy, default = 10
            the standard deviation of the amplitudes drawn for each time/freq bin
    Returns:
        rfi (array-like): shape=(NTIMES,NFREQS)
            a waterfall containing RFI
    """
    if rfi is None:
        rfi = np.zeros((lsts.size, fqs.size), dtype=np.complex)
    assert rfi.shape == (lsts.size, fqs.size), "rfi shape is not (lst.size, fqs.size)"

    bands = np.arange(freq_min, freq_max, width)  # lower freq of each potential DTV band

    # If the bands fit exactly into freqs, the upper band will be the top freq
    # and we need to ignore it.
    if fqs.max() <= bands.max():
        bands = bands[:-1]

    delta_f = fqs[1] - fqs[0]

    chance = _listify(chance)
    strength_std = _listify(strength_std)
    strength = _listify(strength)

    if len(chance) == 1:
        chance *= len(bands)
    if len(strength) == 1:
        strength *= len(bands)
    if len(strength_std) == 1:
        strength_std *= len(bands)

    if len(chance) != len(bands):
        raise ValueError("chance must be float or list with len equal to number of bands")
    if len(strength) != len(bands):
        raise ValueError("strength must be float or list with len equal to number of bands")
    if len(strength_std) != len(bands):
        raise ValueError("strength_std must be float or list with len equal to number of bands")

    for band, chnc, strngth, str_std in zip(bands, chance, strength, strength_std):
        fq_ind_min = np.argwhere(band <= fqs)[0][0]
        fq_ind_max = fq_ind_min + int(width / delta_f) + 1
        this_rfi = rfi[:, fq_ind_min:min(fq_ind_max, fqs.size)]

        rfis = np.random.uniform(size=lsts.size) <= chnc
        this_rfi[rfis] += np.atleast_2d(np.random.normal(strngth, str_std, size=np.sum(rfis)) * np.exp(
            2 * np.pi * 1j * np.random.uniform(size=np.sum(rfis))
        )).T

    return rfi


def _listify(x):
    """
    Ensure a scalar/list is returned as a list.

    Gotten from https://stackoverflow.com/a/1416677/1467820
    """
    if isinstance(x, basestring):
        return [x]
    else:
        try:
            iter(x)
        except TypeError:
            return [x]
        else:
            return list(x)
