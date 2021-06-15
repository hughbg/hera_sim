"""Re-imagining of the simulation module."""

import functools
import inspect
import warnings
import yaml
import time
from pathlib import Path
from typing import Optional, Union, Dict, Tuple, Type, List
from deprecation import deprecated

import numpy as np
from pyuvdata import UVData
from astropy import constants as const

from . import io
from . import utils
from .defaults import defaults
from . import __version__
from .components import SimulationComponent, get_model, list_all_components

_add_depr = deprecated(
    deprecated_in="1.0", removed_in="2.0", details="Use the :meth:`add` method instead."
)


# wrapper for the run_sim method, necessary for part of the CLI
def _generator_to_list(func, *args, **kwargs):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        result = list(func(*args, **kwargs))
        return None if result == [] else result

    return new_func


# FIXME: some of the code in here is pretty brittle and breaks (sometimes silently)
# if not used carefully. This definitely needs a review and cleanup.
class Simulator:
    """Class for managing a simulation.

    Parameters
    ----------
    data
        Any input data. If a path, will load a set of
        visibilities using the ``read()`` method of
        :class:`pyuvdata.UVData`. Can be a :class:`UVData` object
        to initialize directly. If not set, initialize an
        empty :class:`pyuvdata.UVData` object.
    defaults_config
        If given, either a path pointing to a defaults configuration
        file, a string identifier of a particular config (e.g. 'h1c')
        or a dictionary of configuration parameters (see :class:`~.defaults.Defaults`).
    **kwargs
        Used to initialize the data object. If nothing is given for ``data``, the
        relevant parameters are those in :func:`~.io.empty_uvdata`. If ``data`` is a
        path, parameters are those passed to `:func:`pyuvdata.UVData.read`.
    """

    def __init__(
        self,
        data: Optional[Union[str, Path, UVData]] = None,
        defaults_config: Optional[Union[str, dict]] = None,
        **kwargs,
    ):
        # create some utility dictionaries
        self._components = {}
        self.extras = {}
        self._seeds = {}
        self._antpairpol_cache = {}

        # apply and activate defaults if specified
        if defaults_config:
            self.apply_defaults(defaults_config)

        # actually initialize the UVData object stored in self.data
        self._initialize_data(data, **kwargs)

    @property
    def antpos(self) -> Dict[int, Tuple[float, float, float]]:
        """Dictionary of antenna numbers mapped to ENU positions."""
        antpos, ants = self.data.get_ENU_antpos(pick_data_ants=True)
        return dict(zip(ants, antpos))

    @property
    def lsts(self) -> np.ndarray:
        """Unique LSTs in the data array."""
        return np.unique(self.data.lst_array)

    @property
    def freqs(self) -> np.ndarray:
        """Frequencies in GHz."""
        return np.unique(self.data.freq_array) / 1e9

    @property
    def times(self) -> np.ndarray:
        """Unique simulation JDs."""
        return np.unique(self.data.time_array)

    def apply_defaults(self, config: Optional[Union[str, dict]], refresh: bool = True):
        """Apply a given set of defaults.

        Parameters
        ----------
        config
            If given, either a path pointing to a defaults configuration
            file, a string identifier of a particular config (e.g. 'h1c')
            or a dictionary of configuration parameters
            (see :class:`~.defaults.Defaults`).
        refresh
            Whether to refresh the defaults.
        """
        defaults.set(config, refresh=refresh)

    def add(
        self,
        component: Union[SimulationComponent, Type[SimulationComponent], str],
        **kwargs,
    ) -> Optional[UVData]:
        """Add a particular simulation component to the simulated data.

        Parameters
        ----------
        component
            Either a string name (or alias) of a component model to add,
            or its actual class, or an instance of the class.

        Other Parameters
        ----------------
        add_vis
            Whether to add the simulated component visibilities to the
            the ``data``. Sometimes you might just want to return the
            particular component without adding it to the data. Default is True.
        ret_vis
            Whether to return the simulated component. Useful if ``add_vis=False``,
            or you just want a reference to this particular component without being
            mixed with the rest of the simulation.

        Returns
        -------
        Optional[UVData]
            If ``ret_vis=True``, the data simulated from this component.
        """
        # find out whether to add and/or return the component
        add_vis = kwargs.pop("add_vis", True)
        ret_vis = kwargs.pop("ret_vis", False)

        # find out whether the data application should be filtered
        vis_filter = kwargs.pop("vis_filter", None)

        # take out the seed kwarg so as not to break initializor
        seed = kwargs.pop("seed", -1)

        # get the model for the desired component
        model, is_class = self._get_component(component)

        # make a new entry in the antpairpol cache
        self._antpairpol_cache[model] = []

        # get a reference to the cache
        antpairpol_cache = self._antpairpol_cache[model]

        # make sure to keep the key handy in case it's a class
        model_key = model

        # instantiate the class if the component is a class
        if is_class:
            model = model(**kwargs)

        # check that there isn't an issue with component ordering
        self._sanity_check(model)

        # re-add the seed kwarg if it was specified
        if seed != -1:
            kwargs["seed"] = seed

        # calculate the effect
        data = self._iteratively_apply(
            model,
            add_vis=add_vis,
            ret_vis=ret_vis,
            vis_filter=vis_filter,
            antpairpol_cache=antpairpol_cache,
            **kwargs,
        )

        # log the component and its kwargs, if added to data
        if add_vis:
            # note the filter used if any
            if vis_filter is not None:
                kwargs["vis_filter"] = vis_filter
            # note the defaults used if any
            if defaults._override_defaults:
                kwargs["defaults"] = defaults()
            # log the component and the settings used
            self._components[component] = kwargs
            # update the history
            self._update_history(model, **kwargs)
            # track the seed(s) used, if any
            if seed != -1:
                self._update_seeds(self._get_model_name(model))
        else:
            # if we're not adding it, then we don't want to keep
            # the antpairpol cache
            _ = self._antpairpol_cache.pop(model_key)

        # return the data if desired
        if ret_vis:
            return data

    def get(
        self,
        component: str,
        ant1: Optional[int] = None,
        ant2: Optional[int] = None,
        pol: Optional[str] = None,
    ) -> np.ndarray:
        """Obtain a particular simulation component that has already been simulated.

        Parameters
        ----------
        component
            The name of the component to re-simulate.
        ant1
            The antenna number to obtain. If None, both ``ant1`` and ``ant2``
            must be None, and data for all antenna pairs will be returned.
        ant2
            The antenna number to obtain. If None, both ``ant1`` and ``ant2``
            must be None, and data for all antenna pairs will be returned.
        pol
            Polarization to obtain. If None, all pols will be returned.

        Returns
        -------
        ndarray
            Visibilities of this component, shape ``(blt, freq, pol)``.

        Raises
        ------
        AttributeError
            If component has not been simulated before.
        TypeError
            If either ``ant1`` or ``ant2`` is specified but not both.
        """
        # TODO: figure out if this could be handled by _iteratively_apply
        # TODO: determine whether to leave this check here.
        if component not in self._components:
            raise AttributeError(
                "You are trying to retrieve a component that has not "
                "been simulated. Please check that the component you "
                "are passing is correct. Consult the _components "
                "attribute to see which components have been simulated "
                "and which keys are provided."
            )

        if (ant1 is None) ^ (ant2 is None):
            raise TypeError(
                "You are trying to retrieve a visibility but have only "
                "specified one antenna. This use is unsupported; please "
                "either specify an antenna pair or leave both as None."
            )

        # retrieve the model
        model, is_class = self._get_component(component)

        # get the kwargs
        kwargs = self._components[component].copy()

        # figure out whether or not to seed the rng
        seed = kwargs.pop("seed", None)

        # get the antpairpol cache
        antpairpol_cache = self._antpairpol_cache[model]

        # figure out whether or not to apply defaults
        use_defaults = kwargs.pop("defaults", {})
        if use_defaults:
            self.apply_defaults(use_defaults)

        # instantiate the model if it's a class
        if is_class:
            model = model(**kwargs)

        # if ant1, ant2 not specified, then do the whole array
        # TODO: proofread this to make sure that seeds are handled correctly
        if ant1 is None and ant2 is None:
            # re-add seed to the kwargs
            kwargs["seed"] = seed

            # get the data
            data = self._iteratively_apply(
                model,
                add_vis=False,
                ret_vis=True,
                antpairpol_cache=antpairpol_cache,
                **kwargs,
            )

            # return a subset if a polarization is specified
            if pol is None:
                return data
            pol_ind = self.data.get_pols().index(pol)
            return data[:, 0, :, pol_ind]

        # seed the RNG if desired, but be careful...
        if seed == "once":
            # in this case, we need to use _iteratively_apply
            # otherwise, the seeding will be wrong
            kwargs["seed"] = seed
            data = self._iteratively_apply(model, add_vis=False, ret_vis=True, **kwargs)
            blt_inds = self.data.antpair2ind((ant1, ant2), ordered=False)
            if pol is None:
                return data[blt_inds, 0, :, :]
            pol_ind = self.data.get_pols().index(pol)
            return data[blt_inds, 0, :, pol_ind]
        elif seed == "redundant":
            # TODO: this will need to be modified when polarization handling is fixed
            # putting the comment here for lack of a "best" place to put it
            if any((ant2, ant1) == item[:-1] for item in antpairpol_cache):
                self._seed_rng(seed, model, ant2, ant1)
            else:
                self._seed_rng(seed, model, ant1, ant2)

        # get the arguments necessary for the model
        args = self._initialize_args_from_model(model)
        args = self._update_args(args, ant1, ant2, pol)
        args.update(kwargs)

        # now calculate the effect and return it
        return model(**args)

    def plot_array(self):
        """Generate a plot of the array layout in ENU coordinates."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("East Position [m]", fontsize=12)
        ax.set_ylabel("North Position [m]", fontsize=12)
        ax.set_title("Array Layout", fontsize=12)
        dx = 0.25
        for ant, pos in self.antpos.items():
            ax.plot(pos[0], pos[1], color="k", marker="o")
            ax.text(pos[0] + dx, pos[1] + dx, ant)
        return fig

    def refresh(self):
        """Refresh the object.

        This zeros the data array, resets the history, and clears the
        instance's _components dictionary.
        """
        self.data.data_array = np.zeros(self.data.data_array.shape, dtype=complex)
        self.data.history = ""
        self._components.clear()
        self._antpairpol_cache = {}

    def write(self, filename, save_format="uvh5", **kwargs):
        """Write the ``data`` to file in given format."""
        try:
            getattr(self.data, f"write_{save_format}")(filename, **kwargs)
        except AttributeError:
            raise ValueError(
                "The save_format must correspond to a write method in UVData."
            )

    # TODO: Determine if we want to provide the user the option to retrieve
    # simulation components as a return value from run_sim. Remove the
    # _generator_to_list wrapper if we do not make that a feature.
    @_generator_to_list
    def run_sim(
        self, sim_file: Optional[Union[str, Path]] = None, **sim_params
    ) -> Optional[List[np.ndarray]]:
        """Run a full simulation.

        Parameters
        ----------
        sim_file
            Path to a YAML file containing configuration of the simulation components.
        **sim_params
            If ``sim_file`` is not given, configuration can be provided directly as
            parameters.

        Returns
        -------
        components
            If ``ret_vis=True``, the visibilities from each added simulation component
            as a list.
        """
        # TODO: fill out docstring more comprehensively

        # make sure that only sim_file or sim_params are specified
        if not (bool(sim_file) ^ bool(sim_params)):
            raise ValueError(
                "Either an absolute path to a simulation configuration "
                "file or a dictionary of simulation parameters may be "
                "passed, but not both. Please only pass one of the two."
            )

        # read the simulation file if provided
        if sim_file is not None:
            with open(sim_file, "r") as config:
                try:
                    sim_params = yaml.load(config.read(), Loader=yaml.FullLoader)
                except Exception:
                    raise IOError("The configuration file was not able to be loaded.")

        # loop over the entries in the configuration dictionary
        for component, params in sim_params.items():
            # make sure that the parameters are a dictionary
            if not isinstance(params, dict):
                raise TypeError(
                    "The parameters for {component} are not formatted "
                    "properly. Please ensure that the parameters for "
                    "each component are specified using a "
                    "dictionary.".format(component=component)
                )

            # add the component to the data
            value = self.add(component, **params)

            # if the user wanted to return the data, then
            if value is not None:
                yield component, value

    def chunk_sim_and_save(
        self,
        save_dir,
        ref_files=None,
        Nint_per_file=None,
        prefix=None,
        sky_cmp=None,
        state=None,
        filetype="uvh5",
        clobber=True,
    ):
        """
        Chunk a simulation in time and write to disk.

        This function is a thin wrapper around :func:`~.io.chunk_sim_and_save`;
        please see that function's documentation for more information.
        """
        io.chunk_sim_and_save(
            self.data,
            save_dir,
            ref_files=ref_files,
            Nint_per_file=Nint_per_file,
            prefix=prefix,
            sky_cmp=sky_cmp,
            state=state,
            filetype=filetype,
            clobber=clobber,
        )

    # -------------- Legacy Functions -------------- #
    @_add_depr
    def add_eor(self, model, **kwargs):
        """Add an EoR-like model to the visibilities."""
        return self.add(model, **kwargs)

    @_add_depr
    def add_foregrounds(self, model, **kwargs):
        """Add foregrounds to the visibilities."""
        return self.add(model, **kwargs)

    @_add_depr
    def add_noise(self, model, **kwargs):
        """Add thermal noise to the visibilities."""
        return self.add(model, **kwargs)

    def _get_reds(self):
        return self.data.get_redundancies()[0]

    @_add_depr
    def add_rfi(self, model, **kwargs):
        """Add RFI to the visibilities."""
        return self.add(model, **kwargs)

    @_add_depr
    def add_gains(self, **kwargs):
        """Apply bandpass gains to the visibilities."""
        return self.add("gains", **kwargs)

    @_add_depr
    def add_sigchain_reflections(self, ants=None, **kwargs):
        """Apply reflection gains to the visibilities."""
        kwargs.update(ants=ants)
        return self.add("reflections", **kwargs)

    @_add_depr
    def add_xtalk(self, model="gen_whitenoise_xtalk", bls=None, **kwargs):
        """Add crosstalk to the visibilities."""
        kwargs.update(vis_filter=bls)
        return self.add(model, **kwargs)

    @staticmethod
    def _apply_filter(vis_filter, ant1, ant2, pol):
        # TODO: docstring
        # find out whether or not multiple keys are passed
        multikey = any(isinstance(key, (list, tuple)) for key in vis_filter)
        # iterate over the keys, find if any are okay
        if multikey:
            apply_filter = [
                Simulator._apply_filter(key, ant1, ant2, pol) for key in vis_filter
            ]
            # if a single filter says to let it pass, then do so
            return all(apply_filter)
        elif all(item is None for item in vis_filter):
            # support passing tuple of None
            return False
        elif len(vis_filter) == 1:
            # check if the polarization matches, since the only
            # string identifiers should be polarization strings
            if isinstance(vis_filter, str):
                return not pol == vis_filter[0]
            # otherwise assume that this is specifying an antenna
            else:
                return not vis_filter[0] in (ant1, ant2)
        elif len(vis_filter) == 2:
            # there are three cases: two polarizations are specified;
            # an antpol is specified; a baseline is specified
            # first, handle the case of two polarizations
            if all(isinstance(key, str) for key in vis_filter):
                return pol not in vis_filter
            # otherwise it's simple
            else:
                return not all(key in (ant1, ant2, pol) for key in vis_filter)
        elif len(vis_filter) == 3:
            # assume it's a proper antpairpol
            return not (
                vis_filter == [ant1, ant2, pol] or vis_filter == [ant2, ant1, pol]
            )
        else:
            # assume it's some list of antennas/polarizations
            return not any(key in (ant1, ant2, pol) for key in vis_filter)

    def _initialize_data(self, data, **kwargs):
        # TODO: docstring
        if data is None:
            self.data = io.empty_uvdata(**kwargs)
        elif isinstance(data, (str, Path)):
            self.data = self._read_datafile(data, **kwargs)
            self.extras["data_file"] = data
        elif isinstance(data, UVData):
            self.data = data
        else:
            raise TypeError("Unsupported type.")  # make msg better

    def _initialize_args_from_model(self, model):
        # TODO: docstring
        model_params = self._get_model_parameters(model)
        _ = model_params.pop("kwargs", None)

        # pull the lst and frequency arrays as required
        args = {
            param: getattr(self, param)
            for param in model_params
            if param in ("lsts", "freqs")
        }

        model_params.update(args)

        return model_params

    def _iterate_antpair_pols(self):
        # TODO: docstring
        for ant1, ant2, pol in self.data.get_antpairpols():
            blt_inds = self.data.antpair2ind((ant1, ant2))
            pol_ind = self.data.get_pols().index(pol)
            yield ant1, ant2, pol, blt_inds, pol_ind

    def _iteratively_apply(
        self,
        model,
        add_vis=True,
        ret_vis=False,
        vis_filter=None,
        antpairpol_cache=None,
        **kwargs,
    ):
        # TODO: docstring
        # do nothing if neither adding nor returning the effect
        if not add_vis and not ret_vis:
            warnings.warn(
                "You have chosen to neither add nor return the effect "
                "you are trying to simulate, so nothing will be "
                "computed. This warning was raised for the model: "
                "{model}".format(model=self._get_model_name(model))
            )
            return

        # make an empty list for antpairpol cache if it's none
        if antpairpol_cache is None:
            antpairpol_cache = []

        # pull lsts/freqs if required and find out which extra
        # parameters are required
        args = self._initialize_args_from_model(model)

        # figure out whether or not to seed the RNG
        seed = kwargs.pop("seed", None)

        # get a copy of the data array
        data_copy = self.data.data_array.copy()

        # find out if the model is multiplicative
        is_multiplicative = getattr(model, "is_multiplicative", None)

        # handle user-defined functions as the passed model
        if is_multiplicative is None:
            warnings.warn(
                "You are attempting to compute a component but have "
                "not specified an ``is_multiplicative`` attribute for "
                "the component. The component will be added under "
                "the assumption that it is *not* multiplicative."
            )
            is_multiplicative = False

        for ant1, ant2, pol, blt_inds, pol_ind in self._iterate_antpair_pols():
            # find out whether or not to filter the result
            apply_filter = self._apply_filter(
                utils._listify(vis_filter), ant1, ant2, pol
            )

            # check if the antpolpair or its conjugate have data
            bl_in_cache = (ant1, ant2, pol) in antpairpol_cache
            conj_in_cache = (ant2, ant1, pol) in antpairpol_cache

            if seed == "redundant" and conj_in_cache:
                seed = self._seed_rng(seed, model, ant2, ant1)
            elif seed is not None:
                seed = self._seed_rng(seed, model, ant1, ant2)

            # parse the model signature to get the required arguments
            use_args = self._update_args(args, ant1, ant2, pol)

            # update with the passed kwargs
            use_args.update(kwargs)

            # if neither are in the cache, then add it to the cache
            if not (bl_in_cache or conj_in_cache):
                antpairpol_cache.append((ant1, ant2, pol))

            # check whether we're simulating a gain or a visibility
            if is_multiplicative:
                # get the gains for the entire array
                # this is sloppy, but ensures seeding works correctly
                gains = model(**use_args)

                # now get the product g_1g_2*
                gain = gains[ant1] * np.conj(gains[ant2])

                # don't actually do anything if we're filtering this
                if apply_filter:
                    gain = np.ones(gain.shape)

                # apply the effect to the appropriate part of the data
                data_copy[blt_inds, 0, :, pol_ind] *= gain
            else:
                # if the conjugate baseline has been simulated and
                # the RNG was only seeded initially, then we should
                # not re-simulate to ensure invariance under complex
                # conjugation and swapping antennas
                if conj_in_cache and seed is None:
                    conj_blts = self.data.antpair2ind((ant2, ant1))
                    vis = (data_copy - self.data.data_array)[
                        conj_blts, 0, :, pol_ind
                    ].conj()
                else:
                    vis = model(**use_args)

                # filter what's actually having data simulated
                if apply_filter:
                    vis = np.zeros(vis.shape, dtype=complex)

                # and add it in
                data_copy[blt_inds, 0, :, pol_ind] += vis

        # return the component if desired
        # this is a little complicated, but it's done this way so that
        # there aren't *three* copies of the data array floating around
        # this is to minimize the potential of triggering a MemoryError
        if ret_vis:
            data_copy -= self.data.data_array
            # the only time we're allowed to have add_vis be False is
            # if ret_vis is True, and nothing happens if both are False
            # so this is the *only* case where we'll have to reset the
            # data array
            if add_vis:
                self.data.data_array += data_copy
            # return the gain dictionary if gains are simulated
            if is_multiplicative:
                return gains
            # otherwise return the actual visibility simulated
            else:
                return data_copy
        else:
            self.data.data_array = data_copy

    @staticmethod
    def _read_datafile(datafile: Union[str, Path], **kwargs) -> UVData:
        """Read a file as a ``UVData`` object.

        Parameters
        ----------
        datafile
            Path to a file containing visibility data readable by ``pyuvdata``.
        **kwargs
            Arguments passed to the ``UVData.read`` method.

        Returns
        -------
        UVData
            The read-in data object.
        """
        uvd = UVData()
        uvd.read(datafile, read_data=True, **kwargs)
        return uvd

    def _seed_rng(self, seed, model, ant1=None, ant2=None):
        """Seed the random number generator."""
        # TODO: fill out docstring.
        if not isinstance(seed, str):
            raise TypeError("The seeding mode must be specified as a string.")
        if seed == "redundant":
            if ant1 is None or ant2 is None:
                raise TypeError(
                    "A baseline must be specified in order to "
                    "seed by redundant group."
                )

            # generate seeds for each redundant group
            # this does nothing if the seeds already exist
            self._generate_redundant_seeds(model)

            # Determine the key for the redundant group this baseline is in.
            bl_int = self.data.antnums_to_baseline(ant1, ant2)
            red_grps = self._get_reds()
            key = next(reds for reds in red_grps if bl_int in reds)[0]
            # seed the RNG accordingly
            np.random.seed(self._get_seed(model, key))
            return "redundant"
        elif seed == "once":
            # this option seeds the RNG once per iteration of
            # _iteratively_apply, using the same seed every time
            # this is appropriate for antenna-based gains (where the
            # entire gain dictionary is simulated each time), or for
            # something like PointSourceForeground, where objects on
            # the sky are being placed randomly
            np.random.seed(self._get_seed(model, 0))
            return "once"
        elif seed == "initial":
            # this seeds the RNG once at the very beginning of
            # _iteratively_apply. this would be useful for something
            # like ThermalNoise
            np.random.seed(self._get_seed(model, -1))
            return None
        else:
            raise ValueError("Seeding mode not supported.")

    def _update_args(self, args, ant1=None, ant2=None, pol=None):
        # TODO: docstring
        # helper for getting the correct parameter name
        def key(requires):
            return list(args)[requires.index(True)]

        # find out what needs to be added to args
        # for antenna-based gains
        _requires_ants = [param.startswith("ant") for param in args]
        requires_ants = any(_requires_ants)
        # for sky components
        _requires_bl_vec = [param.startswith("bl") for param in args]
        requires_bl_vec = any(_requires_bl_vec)
        # for cross-coupling xtalk
        _requires_vis = [param.find("vis") != -1 for param in args]
        requires_vis = any(_requires_vis)

        # check if this is an antenna-dependent quantity; should
        # only ever be true for gains (barring future changes)
        if requires_ants:
            new_param = {key(_requires_ants): self.antpos}
        # check if this is something requiring a baseline vector
        # current assumption is that these methods require the
        # baseline vector to be provided in nanoseconds
        elif requires_bl_vec:
            bl_vec = self.antpos[ant2] - self.antpos[ant1]
            bl_vec_ns = bl_vec * 1e9 / const.c.value
            new_param = {key(_requires_bl_vec): bl_vec_ns}
        # check if this is something that depends on another
        # visibility. as of now, this should only be cross coupling
        # crosstalk
        elif requires_vis:
            autovis = self.data.get_data(ant1, ant1, pol)
            new_param = {key(_requires_vis): autovis}
        else:
            new_param = {}
        # update appropriately and return
        use_args = args.copy()
        use_args.update(new_param)

        # there should no longer be any unspecified, required parameters
        # so this *shouldn't* error out
        use_args = {
            key: value
            for key, value in use_args.items()
            if not type(value) is inspect.Parameter
        }

        if any([val is inspect._empty for val in use_args.values()]):
            warnings.warn(
                "One of the required parameters was not extracted. "
                "Please check that the parameters for the model you "
                "are trying to add are detectable by the Simulator. "
                "The Simulator will automatically find the following "
                "required parameters: \nlsts \nfreqs \nAnything that "
                "starts with 'ant' or 'bl'\n Anything containing 'vis'."
            )

        return use_args

    @staticmethod
    def _get_model_parameters(model):
        """Retrieve the full model signature (init + call) parameters."""
        init_params = inspect.signature(model.__class__).parameters
        call_params = inspect.signature(model).parameters
        # this doesn't work correctly if done on one line
        model_params = {}
        model_params.update(**call_params, **init_params)
        _ = model_params.pop("kwargs", None)
        return model_params

    @staticmethod
    def _get_component(
        component: Union[str, Type[SimulationComponent], SimulationComponent]
    ) -> Tuple[Union[SimulationComponent, Type[SimulationComponent]], bool]:
        """Normalize a component to be either a class or instance."""
        if np.issubclass_(component, SimulationComponent):
            return component, True
        elif isinstance(component, str):
            try:
                return get_model(component), True
            except KeyError:
                raise ValueError(
                    f"The model '{component}' does not exist. The following models are "
                    f"available: \n{list_all_components()}."
                )
        elif isinstance(component, SimulationComponent):
            return component, False
        else:
            raise ValueError(
                "The input type for the component was not understood. "
                "Must be a string, or a class/instance of type 'SimulationComponent'. "
                f"Available component models are:\n{list_all_components()}"
            )

    def _generate_seed(self, model, key):
        # TODO: docstring
        model = self._get_model_name(model)
        # for the sake of randomness
        np.random.seed(int(time.time() * 1e6) % 2 ** 32)
        if model not in self._seeds:
            self._seeds[model] = {}
        self._seeds[model][key] = np.random.randint(2 ** 32)

    def _generate_redundant_seeds(self, model):
        # TODO: docstring
        model = self._get_model_name(model)
        if model in self._seeds:
            return
        for red_grp in self._get_reds():
            self._generate_seed(model, red_grp[0])

    def _get_seed(self, model, key):
        # TODO: docstring
        model = self._get_model_name(model)
        if model not in self._seeds:
            self._generate_seed(model, key)
        # TODO: handle conjugate baselines here instead of other places
        if key not in self._seeds[model]:
            self._generate_seed(model, key)
        return self._seeds[model][key]

    @staticmethod
    def _get_model_name(model):
        # TODO: docstring
        if isinstance(model, str):
            return model
        elif np.issubclass_(model, SimulationComponent):
            return model.__name__
        elif isinstance(model, SimulationComponent):
            return model.__class__.__name__
        else:
            raise TypeError(
                "You are trying to simulate an effect using a custom function. "
                "Please refer to the tutorial for instructions regarding how "
                "to define new simulation components compatible with the Simulator."
            )

    def _sanity_check(self, model):
        # TODO: docstring
        has_data = not np.all(self.data.data_array == 0)
        is_multiplicative = getattr(model, "is_multiplicative", False)
        contains_multiplicative_effect = any(
            self._get_component(component)[0].is_multiplicative
            for component in self._components
        )

        if is_multiplicative and not has_data:
            warnings.warn(
                "You are trying to compute a multiplicative "
                "effect, but no visibilities have been "
                "simulated yet."
            )
        elif not is_multiplicative and contains_multiplicative_effect:
            warnings.warn(
                "You are adding visibilities to a data array "
                "*after* multiplicative effects have been "
                "introduced."
            )

    def _update_history(self, model, **kwargs):
        # TODO: docstring
        component = self._get_model_name(model)
        msg = f"hera_sim v{__version__}: Added {component} using kwargs:\n"
        if defaults._override_defaults:
            kwargs["defaults"] = defaults._config_name
        for param, value in defaults._unpack_dict(kwargs).items():
            msg += f"{param} = {value}\n"
        self.data.history += msg

    def _update_seeds(self, model_name=None):
        """Update the seeds in the extra_keywords property."""
        seed_dict = {}
        for component, seeds in self._seeds.items():
            if model_name is not None and component != model_name:
                continue

            if len(seeds) == 1:
                seed = list(seeds.values())[0]
                key = "_".join([component, "seed"])
                seed_dict[key] = seed
            else:
                # This should only be raised for seeding by redundancy.
                # Each redundant group is denoted by the *first* baseline
                # integer for the particular redundant group. See the
                # _generate_redundant_seeds method for reference.
                for bl_int, seed in seeds.items():
                    key = "_".join([component, "seed", str(bl_int)])
                    seed_dict[key] = seed

        # Now actually update the extra_keywords dictionary.
        self.data.extra_keywords.update(seed_dict)
