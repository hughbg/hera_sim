"""A module for simulating EoR-like visibilities.

EoR models should require lsts, frequencies, and a baseline vector as
arguments, and may have arbitrary optional parameters. Models should
return complex-valued arrays with shape (Nlsts, Nfreqs) that represent
a visibility appropriate for the given baseline.
"""

import numpy as np
from .components import registry
from . import utils


@registry
class EoR:
    pass


class NoiselikeEoR(EoR):
    """Generate a noiselike, fringe-filtered EoR visibility.

    Parameters
    ----------
    eor_amp : float

    """

    _alias = ("noiselike_eor",)

    def __init__(
        self,
        eor_amp=1e-5,
        min_delay=None,
        max_delay=None,
        fringe_filter_type="tophat",
        fringe_filter_kwargs={},
    ):
        # TODO: docstring
        """
        """
        super().__init__(
            eor_amp=eor_amp,
            min_delay=min_delay,
            max_delay=max_delay,
            fringe_filter_type=fringe_filter_type,
            fringe_filter_kwargs=fringe_filter_kwargs,
        )

    def __call__(self, lsts, freqs, bl_vec, **kwargs):
        # validate the kwargs
        self._check_kwargs(**kwargs)

        # unpack the kwargs
        (
            eor_amp,
            min_delay,
            max_delay,
            fringe_filter_type,
            fringe_filter_kwargs,
        ) = self._extract_kwarg_values(**kwargs)

        # make white noise in freq/time
        # XXX: original says in frate/freq, not sure why
        data = utils.gen_white_noise(size=(len(lsts), len(freqs)))

        # scale data by EoR amplitude
        data *= eor_amp

        # apply delay filter; default does nothing
        # XXX find out why bl_len_ns is hardcoded as 1e10
        # XXX also find out why a tophat filter is hardcoded
        data = utils.rough_delay_filter(
            data,
            freqs,
            1e10,
            delay_filter_type="tophat",
            min_delay=min_delay,
            max_delay=max_delay,
        )

        # apply fringe-rate filter
        data = utils.rough_fringe_filter(
            data,
            lsts,
            freqs,
            bl_vec[0],
            fringe_filter_type=fringe_filter_type,
            **fringe_filter_kwargs
        )

        # dirty trick to make autocorrelations real-valued
        # TODO Figure out the statistically correct way to handle autos.
        # Handling autos this way makes the covariance look like it has
        # no structure... which is wrong.
        if np.all(np.isclose(bl_vec, 0)):
            data = data.real.astype(np.complex)

        return data


noiselike_eor = NoiselikeEoR()
