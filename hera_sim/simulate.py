"""Re-imagining of the simulation module."""

import functools
import inspect
import os
import sys
import warnings
import yaml
import time

import numpy as np
from cached_property import cached_property
from pyuvdata import UVData
from astropy import constants as const

from . import io
from . import utils
from .defaults import defaults
from .version import version
from .components import SimulationComponent

# wrapper for the run_sim method, necessary for part of the CLI
def _generator_to_list(func, *args, **kwargs):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        result = list(func(*args, **kwargs))
        return None if result == [] else result
    return new_func

class Simulator:
    """Class for managing a simulation.

    """
    def __init__(self, data=None, defaults_config=None, **kwargs):
        """Initialize a Simulator object.

        Idea: Make Simulator object have three major components:
            sim.data -> UVData object for storing the "measured" data
                Also keep track of most metadata here
            sim.defaults -> Defaults object

        """
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

    @cached_property
    def antpos(self):
        # TODO: docstring
        """
        """
        antpos, ants = self.data.get_ENU_antpos(pick_data_ants=True)
        return dict(zip(ants, antpos))

    @cached_property
    def lsts(self):
        # TODO: docstring
        return np.unique(self.data.lst_array)

    @cached_property
    def freqs(self):
        # TODO: docstring
        """Frequencies in GHz
        """
        return np.unique(self.data.freq_array) / 1e9

    # XXX begin methods intended for user interaction XXX

    def apply_defaults(self, config, refresh=True):
        # TODO: docstring
        """
        """
        # actually apply the default settings
        defaults.set(config, refresh=refresh)

    def add(self, component, **kwargs):
        # TODO: docstring
        """
        """
        # find out whether to add and/or return the component
        add_vis = kwargs.pop("add_vis", True)
        ret_vis = kwargs.pop("ret_vis", False)

        # find out whether the data application should be filtered
        vis_filter = kwargs.pop("vis_filter", None)
        
        # take out the seed_mode kwarg so as not to break initializor
        seed_mode = kwargs.pop("seed_mode", -1)
        
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
        
        # re-add the seed_mode kwarg if it was specified
        if seed_mode != -1:
            kwargs["seed_mode"] = seed_mode
        
        # calculate the effect
        data = self._iteratively_apply(
            model, add_vis=add_vis, ret_vis=ret_vis, 
            vis_filter=vis_filter, 
            antpairpol_cache=antpairpol_cache,
            **kwargs
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
        else:
            # if we're not adding it, then we don't want to keep
            # the antpairpol cache
            _ = self._antpairpol_cache.pop(model_key)
        
        # return the data if desired
        if ret_vis:
            return data

    def get(self, component, ant1=None, ant2=None, pol=None):
        # TODO: docstring
        # XXX ideally, this could be handled by _iteratively_apply
        """
        """
        # XXX do we want to leave this check in there?
        if component not in self._components:
            raise AttributeError(
                "You are trying to retrieve a component that has not " 
                "been simulated. Please check that the component you " 
                "are passing is correct. Consult the _components " 
                "attribute to see which components have been simulated " 
                "and which keys are provided."
            )
            
        if ((ant1 is None) ^ (ant2 is None)):
            raise TypeError(
                "You are trying to retrieve a visibility but have only " 
                "specified one antenna. This use is unsupported; please " 
                "either specify an antenna pair or leave both as None."
            )

        # retrieve the model
        model, is_class = self._get_component(component)
        
        # get the kwargs
        kwargs = self._components[component]
        
        # figure out whether or not to seed the rng
        seed_mode = kwargs.pop("seed_mode", None)
        
        # get the antpairpol cache
        antpairpol_cache = self._antpairpol_cache[model]

        # figure out whether or not to apply defaults
        use_defaults = kwargs.pop("defaults", {})
        if use_defaults:
            self.apply_defaults(**use_defaults)
        
        # instantiate the model if it's a class
        if is_class:
            model = model(**kwargs)
        
        # if ant1, ant2 not specified, then do the whole array
        if ant1 is None and ant2 is None:
            # re-add seed_mode to the kwargs
            kwargs["seed_mode"] = seed_mode

            # get the data
            data = self._iteratively_apply(
                model, add_vis=False, ret_vis=True, 
                antpairpol_cache=antpairpol_cache,
                **kwargs
            )
            
            # return a subset if a polarization is specified
            if pol is None:
                return data
            else:
                pol_ind = self.data.get_pols().index(pol)
                return data[:, 0, :, pol_ind]
        
        # seed the RNG if desired, but be careful...
        if seed_mode == "once":
            # in this case, we need to use _iteratively_apply
            # otherwise, the seeding will be wrong
            kwargs["seed_mode"] = seed_mode
            data = self._iteratively_apply(
                model, add_vis=False, ret_vis=True, **kwargs
            )
            blt_inds = self.data.antpair2ind((ant1, ant2))
            if pol is None:
                return data[blt_inds, 0, :, :]
            else:
                pol_ind = self.data.get_pols().index(pol)
                return data[blt_inds, 0, :, pol_ind]
        elif seed_mode == "redundant":
            if any(
                [(ant2, ant1) == item for item in antpairpol_cache]
            ):
                self._seed_rng(seed_mode, model, ant2, ant1)
            else:
                self._seed_rng(seed_mode, model, ant1, ant2)
        
        # get the arguments necessary for the model
        init_args = self._initialize_args_from_model(model)
        use_args = self._update_args(*init_args, ant1, ant2, pol)

        # now calculate the effect and return it
        return model(*use_args, **kwargs)

    def plot_array(self):
        """Generate a plot of the array layout in ENU coordinates.

        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel("East Position [m]", fontsize=12)
        ax.set_ylabel("North Position [m]", fontsize=12)
        ax.set_title("Array Layout", fontsize=12)
        dx = 0.25
        for ant, pos in self.antpos.items():
            ax.plot(pos[0], pos[1], color='k', marker='o')
            ax.text(pos[0] + dx, pos[1] + dx, ant)
        return fig

    def refresh(self):
        """Refresh the Simulator object.

        This zeros the data array, resets the history, and clears the 
        instance's _components dictionary.
        """
        self.data.data_array = np.zeros(
            self.data.data_array.shape, dtype=np.complex
        )
        self.data.history = ''
        self._components.clear()
        self._antpairpol_cache = []

    def write(self, filename, save_format="uvh5", save_seeds=True, **kwargs):
        # TODO: docstring
        """
        """
        try:
            getattr(self.data, "write_%s" % save_format)(filename, **kwargs)
        except AttributeError:
            msg = "The save_format must correspond to a write method in UVData."
            raise ValueError(msg)
        if save_seeds:
            seed_file = os.path.splitext(filename)[0] + "_seeds"
            np.save(seed_file, self._seeds)

    @_generator_to_list
    def run_sim(self, sim_file=None, **sim_params):
        # TODO: docstring
        """
        """
        # make sure that only sim_file or sim_params are specified
        if not (bool(sim_file) ^ bool(sim_params)):
            raise ValueError(
                "Either an absolute path to a simulation configuration " 
                "file or a dictionary of simulation parameters may be " 
                "passed, but not both. Please only pass one of the two."
            )

        # read the simulation file if provided
        if sim_file is not None:
            with open(sim_file, 'r') as config:
                try:
                    sim_params = yaml.load(
                        config.read(), Loader=yaml.FullLoader
                    )
                except:
                    print("The configuration file was not able to be loaded.")
                    print("Please fix the file and try again.")
                    sys.exit()

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
                yield (component, value)

    # XXX end methods intended for user interaction XXX

    # XXX begin helper methods XXX

    @staticmethod
    def _apply_filter(vis_filter, ant1, ant2, pol):
        # TODO: docstring
        """
        """
        # find out whether or not multiple keys are passed
        multikey = any(
            [isinstance(key, (list, tuple)) for key in vis_filter]
        )
        # iterate over the keys, find if any are okay
        if multikey:
            apply_filter = [self._apply_filter(key, ant1, ant2, pol)
                            for key in vis_filter]
            # if a single filter says to let it pass, then do so
            return all(apply_filter)
        elif all([item is None for item in vis_filter]):
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
            if all([isinstance(key, str) for key in vis_filter]):
                return not pol in vis_filter
            # otherwise it's simple
            else:
                return not all(
                    [key in (ant1, ant2, pol) for key in vis_filter]
                )
        elif len(vis_filter) == 3:
            # assume it's a proper antpairpol
            return not (
                vis_filter == [ant1, ant2, pol] or
                vis_filter == [ant2, ant1, pol]
            )
        else:
            # assume it's some list of antennas/polarizations
            return not any(
                [key in (ant1, ant2, pol) for key in vis_filter]
            )

    def _initialize_data(self, data, **kwargs):
        # TODO: docstring
        """
        """
        if data is None:
            self.data = io.empty_uvdata(**kwargs)
        elif isinstance(data, str):
            self.data = self._read_datafile(data, **kwargs)
            self.extras['data_file'] = data
        elif isinstance(data, UVData):
            self.data = data
        else:
            raise TypeError("Unsupported type.") # make msg better

    def _initialize_args_from_model(self, model):
        # TODO: docstring
        """
        """
        model_params = inspect.signature(model).parameters
        # pull the lst and frequency arrays as required
        args = list(getattr(self, param) for param in model_params
                    if param in ("lsts", "freqs"))
        # for antenna-based gains
        requires_ants = any([param.startswith("ant")
                             for param in model_params])
        # for sky components
        requires_bl_vec = any([param.startswith("bl") 
                               for param in model_params])
        # for cross-coupling xtalk
        requires_vis = any([param.find("vis") != -1
                            for param in model_params])
        return (args, requires_ants, requires_bl_vec, requires_vis)

    def _iterate_antpair_pols(self):
        # TODO: docstring
        """
        """
        for ant1, ant2, pol in self.data.get_antpairpols():
            blt_inds = self.data.antpair2ind((ant1, ant2))
            pol_ind = self.data.get_pols().index(pol)
            yield ant1, ant2, pol, blt_inds, pol_ind

    def _iteratively_apply(self, model, add_vis=True, ret_vis=False, 
                           vis_filter=None, antpairpol_cache=[], 
                           **kwargs):
        # TODO: docstring
        """
        """
        # do nothing if neither adding nor returning the effect
        if not add_vis and not ret_vis:
            warnings.warn(
                "You have chosen to neither add nor return the effect "
                "you are trying to simulate, so nothing will be "
                "computed. This warning was raised for the model: "
                "{model}".format(model=self._get_model_name(model))
            )
            return
        
        # pull lsts/freqs if required and find out which extra 
        # parameters are required
        (args, requires_ants, requires_bl_vec, 
            requires_vis) = self._initialize_args_from_model(model)
        
        # figure out whether or not to seed the RNG
        seed_mode = kwargs.pop("seed_mode", None)
        
        # get a copy of the data array
        initial_data = self.data.data_array.copy()
        
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
            
            if seed_mode == "redundant":
                if conj_in_cache:
                    self._seed_rng(seed_mode, model, ant2, ant1)
            elif seed_mode is not None:
                seed_mode = self._seed_rng(seed_mode, model, ant1, ant2)
            
            # parse the model signature to get the required arguments
            use_args = self._update_args(
                args, requires_ants, requires_bl_vec, requires_vis,
                ant1, ant2, pol
            )

            # if neither are in the cache, then add it to the cache
            if not (bl_in_cache or conj_in_cache):
                antpairpol_cache.append((ant1, ant2, pol))

            # check whether we're simulating a gain or a visibility
            if is_multiplicative:
                # get the gains for the entire array
                # this is sloppy, but ensures seeding works correctly
                gains = model(*use_args, **kwargs)
                
                # now get the product g_1g_2*
                gain = gains[ant1] * np.conj(gains[ant2])
                
                # don't actually do anything if we're filtering this
                if apply_filter:
                    gain = np.ones(gain.shape)

                # apply the effect to the appropriate part of the data
                self.data.data_array[blt_inds, 0, :, pol_ind] *= gain
            else:
                # if the conjugate baseline has been simulated and 
                # the RNG was only seeded initially, then we should 
                # not re-simulate to ensure invariance under complex
                # conjugation and swapping antennas
                if conj_in_cache and seed_mode is None:
                    conj_blts = sim.data.antpair2ind((ant2,ant1))
                    vis = (
                        self.data.data_array - initial_data
                    )[conj_blts, 0, :, pol_ind].conj()
                else: 
                    vis = model(*use_args, **kwargs)
                
                # filter what's actually having data simulated
                if apply_filter:
                    vis = np.zeros(vis.shape, dtype=np.complex)
                
                # and add it in
                self.data.data_array[blt_inds, 0, :, pol_ind] += vis

        # return the component if desired
        # this is a little complicated, but it's done this way so that
        # there aren't *three* copies of the data array floating around
        # this is to minimize the potential of triggering a MemoryError
        # XXX figure out whether this is a moot effort
        # XXX it is *not* a moot effort! this saves memory!
        if ret_vis:
            initial_data = self.data.data_array - initial_data
            # the only time we're allowed to have add_vis be False is
            # if ret_vis is True, and nothing happens if both are False
            # so this is the *only* case where we'll have to reset the
            # data array
            if not add_vis:
                self.data.data_array -= initial_data
            # return the gain dictionary if gains are simulated
            if is_multiplicative:
                return gains
            # otherwise return the actual visibility simulated
            else:
                return initial_data
        

    @staticmethod
    def _read_datafile(datafile, **kwargs):
        # TODO: docstring
        """
        """
        uvd = UVData()
        uvd.read(datafile, read_data=True, **kwargs)
        return uvd

    def _seed_rng(self, seed_mode, model, ant1=None, ant2=None):
        # TODO: docstring
        """
        """
        if not type(seed_mode) is str:
            raise TypeError(
                "The seeding mode must be specified as a string."
            )
        if seed_mode == "redundant":
            if ant1 is None or ant2 is None:
                raise TypeError(
                    "A baseline must be specified in order to "
                    "seed by redundant group."
                )

            # generate seeds for each redundant group
            # this does nothing if the seeds already exist
            self._generate_redundant_seeds(model)
            # get the baseline integer for baseline (ant1, ant2)
            bl_int = self.data.antnums_to_baseline(ant1, ant2)
            # find out which redundant group the baseline is in
            key = [bl_int in reds 
                   for reds in self._get_reds()].index(True)
            # seed the RNG accordingly
            np.random.seed(self._get_seed(model, key))
            return "redundant"
        elif seed_mode == "once":
            # this option seeds the RNG once per iteration of 
            # _iteratively_apply, using the same seed every time
            # this is appropriate for antenna-based gains (where the 
            # entire gain dictionary is simulated each time), or for
            # something like PointSourceForeground, where objects on
            # the sky are being placed randomly
            np.random.seed(self._get_seed(model, 0))
            return "once"
        elif seed_mode == "initial":
            # this seeds the RNG once at the very beginning of 
            # _iteratively_apply. this would be useful for something
            # like ThermalNoise
            np.random.seed(self._get_seed(model, -1))
            return None
        else:
            raise ValueError("Seeding mode not supported.")

    def _update_args(self, args, requires_ants, requires_bl_vec,
                     requires_vis, ant1=None, ant2=None, pol=None):
        # TODO: docstring
        """
        """
        # check if this is an antenna-dependent quantity; should
        # only ever be true for gains (barring future changes)
        if requires_ants:
            ants = self.antpos
            use_args = args + [ants]
        # check if this is something requiring a baseline vector
        # current assumption is that these methods require the
        # baseline vector to be provided in nanoseconds
        elif requires_bl_vec:
            bl_vec = self.antpos[ant1] - self.antpos[ant2]
            bl_vec_ns = bl_vec * 1e9 / const.c.value
            use_args = args + [bl_vec_ns]
        # check if this is something that depends on another
        # visibility. as of now, this should only be cross coupling
        # crosstalk
        elif requires_vis:
            autovis = self.data.get_data(ant1, ant1, pol)
            use_args = args + [autovis]
        else:
            use_args = args.copy()
        return use_args

    @staticmethod
    def _get_component(component):
        # TODO: docstring
        """
        """
        try:
            if issubclass(component, SimulationComponent):
                # support passing user-defined classes that inherit from
                # the SimulationComponent base class to add method
                return component, True
        except TypeError:
            # this is raised if ``component`` is not a class
            if component.__class__.__name__ == "function":
                raise TypeError(
                    "You are attempting to add a component that is "
                    "modeled using a function. Please convert the "
                    "function to a callable class and try again."
                )
            if callable(component):
                # if it's callable, then it's either a user-defined 
                # function or a class instance
                return component, False
            else:
                if not type(component) is str:
                    # TODO: update this error message to reflect the
                    # change in allowed component types
                    raise TypeError(
                        "``component`` must be either a class which " 
                        "derives from ``SimulationComponent`` or an " 
                        "instance of a callable class, or a function, " 
                        "whose signature is:\n" 
                        "func(lsts, freqs, *args, **kwargs)\n" 
                        "If it is none of the above, then it must be " 
                        "a string which corresponds to the name of a " 
                        "``hera_sim`` class or an alias thereof."i
                    )
                
                # keep track of all known aliases in case desired 
                # component isn't found in the search
                all_aliases = []
                for registry in SimulationComponent.__subclasses__():
                    for model in registry.__subclasses__():
                        aliases = (model.__name__,)
                        aliases += getattr(model, "_alias", ())
                        aliases = [alias.lower() for alias in aliases]
                        for alias in aliases:
                            all_aliases.append(alias)
                        if component.lower() in aliases:
                            return model, True
                
                # if this part is executed, then the model wasn't found, so
                msg = "The component '{component}' wasn't found. The "
                msg += "following aliases are known: \n"
                msg += ", ".join(set(all_aliases))
                msg += "\nPlease ensure that the component you are trying "
                msg += "to add is a subclass of a registry."
                msg = msg.format(component=component)
                raise UnboundLocalError(msg)

    def _generate_seed(self, model, key):
        # TODO: docstring
        """
        """
        model = self._get_model_name(model)
        # for the sake of randomness
        np.random.seed(int(time.time()))
        if model not in self._seeds:
            self._seeds[model] = {}
        self._seeds[model][key] = np.random.randint(2**32)

    def _generate_redundant_seeds(self, model):
        # TODO: docstring
        """
        """
        model = self._get_model_name(model)
        if model in self._seeds:
            return
        for j in range(len(self._get_reds())):
            self._generate_seed(model, j)

    def _get_reds(self):
        # TODO: docstring
        """
        """
        return self.data.get_baseline_redundancies()[0]

    def _get_seed(self, model, key):
        # TODO: docstring
        """
        """
        model = self._get_model_name(model)
        if model not in self._seeds:
            self._generate_seed(model, key)
        if key not in self._seeds[model]:
            self._generate_seed(model, key)
        return self._seeds[model][key]
    
    @staticmethod
    def _get_model_name(model):
        # TODO: docstring
        """
        """
        if isinstance(model, str):
            return model
        try:
            return model.__name__
        except AttributeError:
            # check if it's a user defined function
            if model.__class__.__name__ == "function":
                # don't allow users to pass functions, only classes
                # XXX find out if this check always happens before
                # _get_component is called
                msg = "You are trying to simulate an effect using a "
                msg += "custom function. Please convert your "
                msg += "function into a callable class that inherits "
                msg += "from a registry. To make a registry, simply "
                msg += "define a class and decorate it with the "
                msg += "hera_sim.registry decorator. The registry "
                msg += "does not need to perform any tasks or be "
                msg += "instantiated; it just needs to exist and be "
                msg += "a base class for the custom callable class."
                raise TypeError(msg)
            else:
                return model.__class__.__name__

    def _sanity_check(self, model):
        # TODO: docstring
        """
        """
        has_data = not np.all(self.data.data_array == 0)
        is_multiplicative = getattr(model, "is_multiplicative", False)
        contains_multiplicative_effect = any([
                self._get_component(component)[0].is_multiplicative
                for component in self._components])
        if is_multiplicative and not has_data:
            warnings.warn("You are trying to compute a multiplicative "
                          "effect, but no visibilities have been "
                          "simulated yet.")
        elif not is_multiplicative and contains_multiplicative_effect:
            warnings.warn("You are adding visibilities to a data array "
                          "*after* multiplicative effects have been "
                          "introduced.")

    def _update_history(self, model, **kwargs):
        # TODO: docstring
        """
        """
        model = self._get_model_name(model)
        msg = "hera_sim v{version}: Added {component} using kwargs:\n"
        if defaults._override_defaults:
            kwargs["defaults"] = defaults._config_name
        for param, value in kwargs.items():
            msg += "{param} = {value}\n".format(param=param, value=value)
        msg = msg.format(version=version, component=model)
        self.data.history += msg

