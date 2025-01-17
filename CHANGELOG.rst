=========
Changelog
=========

v1.1.1 [2021.08.21]
===================

Added
-----
- Add a Zernike polynomial beam model.

v1.1.0 [2021.08.04]
===================

Added
-----
- Enable polarization support for ``vis_cpu`` (handles polarized primary beams, but
  only Stokes I sky model so far)
- Add a polarized version of the analytic PolyBeam model.

v1.0.2 [2021.07.01]
===================

Fixed
-----
- Bug in retrieval of unique LSTs by :class:`~.Simulator` when a blt-order other than
  time-baseline is used has been fixed. LSTs should now be correctly retrieved.
- :func:`~.io.empty_uvdata` now sets the ``phase_type`` attribute to "drift".

v1.0.1 [2021.06.30]
===================

Added
-----

Fixed
-----
- Discrepancy in :class:`~.foregrounds.PointSourceForeground` documentation and actual
  implementation has been resolved. Simulated foregrounds now look reasonable.

Changed
-------
- The time parameters used for generating an example ``Simulator`` instance in the tutorial
  have been updated to match their description.
- :class:`~.Simulator` tutorial has been changed slightly to account for the foreground fix.

v1.0.0 [2021.06.16]
===================

Added
-----
- :mod:`~.adjustment` module from HERA Phase 1 Validation work
   - :func:`~.adjustment.adjust_to_reference`
      - High-level interface for making one set of data comply with another set of data.
        This may involve rephasing or interpolating in time and/or interpolating in
        frequency. In the case of a mismatch between the two array layouts, this algorithm
        will select a subset of antennas to provide the greatest number of unique baselines
        that remain in the downselected array.
  - All other functions in this module exist only to modularize the above function.
- :mod:`~.cli_utils` module providing utility functions for the CLI simulation script.
- :mod:`~.components` module providing an abstract base class for simulation components.
   - Any new simulation components should be subclassed from the
     :class:`~.components.SimulationComponent` ABC. New simulation components subclassed
     appropriately are automatically discoverable by the :class:`~.Simulator` class. A MWE
     for subclassing new components is as follows::

        @component
        class Component:
            pass

        class Model(Component):
            ...

     The ``Component`` base class tracks any models subclassed from it and makes it
     discoverable to the :class:`~.Simulator`.
- New "season" configuration (called ``"debug"``), intended to be used for debugging
  the :class:`~.Simulator` when making changes that might not be easily tested.
- :func:`~.io.chunk_sim_and_save` function from HERA Phase 1 Validation work
   - This function allows the user to write a :class:`pyuvdata.UVData` object to disk
     in chunks of some set number of integrations per file (either specified directly,
     or specified implicitly by providing a list of reference files). This is very
     useful for taking a large simulation and writing it to disk in a way that mimics
     how the correlator writes files to disk.
- Ability to generate noise visibilities based on autocorrelations from the data.
  This is achieved by providing a value for the ``autovis`` parameter in
  the ``thermal_noise`` function (see :class:`~.noise.ThermalNoise`).
- The :func:`~.sigchain.vary_gains_in_time` provides an interface for taking a gain
  spectrum and applying time variation (linear, sinusoidal, or noiselike) to any of
  the reflection coefficient parameters (amplitude, phase, or delay).
- The :class:`~.sigchain.CrossCouplingSpectrum` provides an interface for generating
  multiple realizations of the cross-coupling systematic spaced logarithmically in
  amplitude and linearly in delay. This is ported over from the Validation work.

Fixed
-----
- The reionization signal produced by ``eor.noiselike_eor`` is now guaranteed to
  be real-valued for autocorrelations (although the statistics of the EoR signal for
  the autocorrelations still need to be investigated for correctness).

Changed
-------

- **API BREAKING CHANGES**
   - All functions that take frequencies and LSTs as arguments have had their signatures
     changed to ``func(lsts, freqs, *args, **kwargs)``.
   - Functions that employ :func:`~.utils.rough_fringe_filter` or
     :func:`~.utils.rough_delay_filter` as part of the visibility calculation now have
     parameters ``delay_filter_kwargs`` and/or ``fringe_filter_kwargs``, which are
     dictionaries that are ultimately passed to the filtering functions.
     ``foregrounds.diffuse_foreground`` and ``eor.noiselike_eor`` are both affected by this.
   - Some parameters have been renamed to enable simpler handling of package-wide defaults.
     Parameters that have been changed are:
      - ``filter_type`` -> ``delay_filter_type`` in :func:`~.utils.gen_delay_filter`
      - ``filter_type`` -> ``fringe_filter_type`` in :func:`~.utils.gen_fringe_filter`
      - ``chance`` -> ``impulse_chance`` in ``rfi_impulse`` (see :class:`~.rfi.Impulse`)
      - ``strength`` -> ``impulse_strength`` in ``rfi_impulse`` (see :class:`~.rfi.Impulse`)
      - Similar changes were made in ``rfi_dtv`` (:class:`~.rfi.DTV`) and ``rfi_scatter``
        (:class:`~.rfi.Scatter`).
   - Any occurrence of the parameter ``fqs`` has been replaced with ``freqs``.
   - The ``noise.jy2T`` function was moved to :mod:`~.utils` and renamed. See
     :func:`~.utils.jansky_to_kelvin`.
   - The parameter ``fq0`` has been renamed to ``f0`` in :class:`~.rfi.RfiStation`.
   - The ``_listify`` function has been moved from :mod:`~.rfi` to :mod:`~.utils`.
   - ``sigchain.HERA_NRAO_BANDPASS`` no longer exists in the code, but may be loaded from
     the file ``HERA_H1C_BANDPASS.npy`` in the ``data`` directory.
- Other Changes
   - The :class:`~.Simulator` has undergone many changes that make the class much easier
     to use, while also providing a handful of extra features. The new :class:`~.Simulator`
     provides the following features:
      - A universal :meth:`~.Simulator.add` method for applying any of the effects
        implemented in ``hera_sim``, as well as any custom effects defined by the user.
      - A :meth:`~.Simulator.get` method that retrieves any previously simulated effect.
      - The option to apply a simulated effect to only a subset of antennas, baselines,
        and/or polarizations, accessed through using the ``vis_filter`` parameter.
      - Multiple modes of seeding the random state to achieve a higher degree of realism
        than previously available.
      - The :meth:`~.Simulator.calculate_filters` method pre-calculates the fringe-rate
        and delay filters for the entire array and caches the result. This provides a
        marginal-to-modest speedup for small arrays, but can provide a significant
        speedup for very large arrays. Benchmarking results TBD.
      - An instance of the :class:`~.Simulator` may be generated with an empty call to
        the class if any of the season defaults are active (or if the user has provided
        some other sufficiently complete set of default settings).
      - Some of the methods for interacting with the underlying :class:`pyuvdata.UVData`
        object have been exposed to the :class:`~.Simulator` (e.g. ``get_data``).
      - An easy reference to the :func:`~.io.chunk_sim_and_save` function.
   - :mod:`~.foregrounds`, :mod:`~.eor`, :mod:`~.noise`, :mod:`~.rfi`,
     :mod:`~.antpos`, and :mod:`~.sigchain` have been modified to implement the
     features using callable classes. The old functions still exist for
     backwards-compatibility, but moving forward any additions to visibility or
     systematics simulators should be implemented using callable classes and be
     appropriately subclassed from :class:`~.components.SimulationComponent`.
   - :func:`~.io.empty_uvdata` has had almost all of its parameter values set to default as
     ``None``. Additionally, the ``n_freq``, ``n_times``, ``antennas`` parameters are being
     deprecated and will be removed in a future release.
   - :func:`~.noise.white_noise` is being deprecated. This function has been moved to the
     utility module and can be found at :func:`~.utils.gen_white_noise`.

v0.4.0 [2021.05.01]
===================

Added
-----

- New features added to ``vis_cpu``
    - Analytic beam interpolation
        - Instead of gridding the beam and interpolating the grid using splines,
          the beam can be interpolated directly by calling its ``interp`` method.
        - The user specifies this by passing ``use_pixel_beams=False`` to ``vis_cpu``.
    - A simple MPI parallelization scheme
        - Simulation scripts may be run using ``mpirun/mpiexec``
        - The user imports ``mpi4py`` into their script and passes
          ``mpi_comm=MPI.COMM_WORLD`` to vis_cpu
    - New ``PolyBeam`` and ``PerturbedPolyBeam`` analytic beams (classes)
        - Derived from ``pyuvsim.Analytic beam``
        - Based on axisymmetric Chebyshev polynomial fits to the Fagnoni beam.
        - PerturbedPolyBeam is capable of expressing a range of non-redundancy effects,
          including per-beam stretch factors, perturbed sidelobes, and
          ellipticity/rotation.

v0.3.0 [2019.12.10]
===================

Added
-----
- New sub-package ``simulators``
    - ``VisibilitySimulators`` class
        - Provides a common interface to interferometric visibility simulators.
          Users instantiate one of its subclasses and provide input antenna and
          sky scenarios.
        - ``HealVis`` subclass
        - Provides an interface to the ``healvis`` visibility simulator.
    - ``VisCPU`` subclass
        - Provides an interface to the ``viscpu`` visibility simulator.
    - ``conversions`` module
        - Not intended to be interfaced with by the end user; it provides useful
          coordinate transformations for ``VisibilitySimulators``.

v0.2.0 [2019.11.20]
===================

Added
-----
- Command-line Interface
    - Use anywhere with ``hera_sim run [options] INPUT``
    - Tutorial available on readthedocs

- Enhancement of ``run_sim`` method of ``Simulator`` class
   - Allows for each simulation component to be returned
      - Components returned as a list of 2-tuples ``(model_name, visibility)``
      - Components returned by specifying ``ret_vis=True`` in their kwargs

- Option to seed random number generators for various methods
   - Available via the ``Simulator.add_`` methods by specifying the kwarg \
     ``seed_redundantly=True``
   - Seeds are stored in ``Simulator`` object, and may be saved as a ``npy`` \
     file when using the ``Simulator.write_data`` method

- New YAML tag ``!antpos``
   - Allows for antenna layouts to be constructed using ``hera_sim.antpos`` \
     functions by specifying parameters in config file

Fixed
-----

- Changelog formatting for v0.1.0 entry

Changed
-------

- Implementation of ``defaults`` module
   - Allows for semantic organization of config files
   - Parameters that have the same name take on the same value
      - e.g. ``std`` in various ``rfi`` functions only has one value, even if \
        it's specified multiple times

v0.1.0 [2019.08.28]
===================

Added
-----

- New module ``interpolators``
   - Classes intended to be interfaced with by end-users:
      - ``Tsky``
         - Provides an interface for generating a sky temperature \
           interpolation object when provided with a ``.npz`` file \
           and interpolation kwargs.
      - ``Beam``, ``Bandpass``
         - Provides an interface for generating either a ``poly1d`` or \
           ``interp1d`` interpolation object when provided with an \
           appropriate datafile.

- New module ``defaults``
   - Provides an interface which allows the user to dynamically adjust \
     default parameter settings for various ``hera_sim`` functions.

- New module ``__yaml_constructors``
   - Not intended to be interfaced with by the end user; this module just \
     provides a location for defining new YAML tags to be used in conjunction \
     with the ``defaults`` module features and the ``Simulator.run_sim`` method.

- New directory ``config``
   - Provides a location to store configuration files.

Fixed
-----

Changed
-------

- HERA-specific variables had their definitions removed from the codebase.
  Objects storing these variables still exist in the codebase, but their
  definitions now come from loading in data stored in various new files
  added to the ``data`` directory.

v0.0.1
======

- Initial released version
