"""
This tests the Simulator object and associated utilities. It does *not*
check for correctness of individual models, as they should be tested
elsewhere.
"""

import itertools
import shutil
import tempfile
import sys
import os
import yaml

import numpy as np
import pytest

from hera_sim.foregrounds import DiffuseForeground, diffuse_foreground
from hera_sim.noise import HERA_Tsky_mdl
from hera_sim.simulate import Simulator
from hera_sim.antpos import hex_array
from hera_sim import DATA_PATH, CONFIG_PATH
from hera_sim.defaults import defaults
from hera_sim.interpolators import Beam
from pyuvdata import UVData


beamfile = os.path.join(DATA_PATH, "HERA_H1C_BEAM_POLY.npy")
omega_p = Beam(beamfile)
Tsky_mdl = HERA_Tsky_mdl["xx"]

Nfreqs = 10
Ntimes = 20


def create_sim(autos=False, **kwargs):
    return Simulator(
        Nfreqs=Nfreqs,
        start_freq=1e8,
        channel_width=1e8 / 1024,
        Ntimes=Ntimes,
        start_time=2458115.9,
        integration_time=10.7,
        array_layout={0: (20.0, 20.0, 0), 1: (50.0, 50.0, 0)},
        no_autos=not autos,
        **kwargs
    )


@pytest.fixture(scope="function")
def base_sim():
    return create_sim()


@pytest.fixture(scope="function")
def ref_sim(base_sim):
    base_sim.add("noiselike_eor", seed="redundant")
    return base_sim


def test_from_empty(base_sim):
    assert all(
        [
            base_sim.data.Ntimes == Ntimes,
            base_sim.data.Nfreqs == Nfreqs,
            np.all(base_sim.data.data_array == 0),
            base_sim.freqs.size == Nfreqs,
            base_sim.freqs.ndim == 1,
            base_sim.lsts.size == Ntimes,
            base_sim.lsts.ndim == 1,
        ]
    )


def test_initialize_from_defaults():
    with open(CONFIG_PATH / "H1C.yaml") as config:
        defaults = yaml.load(config.read())
    setup = defaults["setup"]
    freq_params = setup["frequency_array"]
    time_params = setup["time_array"]
    array = defaults["telescope"]["array_layout"]
    sim = Simulator(defaults_config="h1c")
    assert all(
        [
            sim.freqs.size == freq_params["Nfreqs"],
            sim.freqs[0] == freq_params["start_freq"] / 1e9,
            sim.lsts.size == time_params["Ntimes"],
            sim.times[0] == time_params["start_time"],
            len(sim.antpos) == len(array),
        ]
    )


def test_add_with_str(base_sim):
    base_sim.add("noiselike_eor")
    assert not np.all(base_sim.data.data_array == 0)


def test_add_with_builtin_class(base_sim):
    base_sim.add(DiffuseForeground, Tsky_mdl=Tsky_mdl, omega_p=omega_p)
    assert not np.all(np.isclose(base_sim.data.data_array, 0))


def test_add_with_class_instance(base_sim):
    base_sim.add(diffuse_foreground, Tsky_mdl=Tsky_mdl, omega_p=omega_p)
    assert not np.all(np.isclose(base_sim.data.data_array, 0))


def test_refresh(base_sim):
    base_sim.add("noiselike_eor")
    base_sim.refresh()

    assert np.all(base_sim.data.data_array == 0)


@pytest.mark.parametrize("make_sim_from", ["uvdata", "file"])
def test_io(base_sim, make_sim_from, tmp_path):
    # Simulate some data and write it to disk.
    filename = tmp_path / "tmp_data.uvh5"
    base_sim.add("pntsrc_foreground")
    base_sim.add("gains")
    base_sim.write(filename)

    if make_sim_from == "file":
        sim2 = Simulator(data=filename)
    elif make_sim_from == "uvdata":
        uvd = UVData()
        uvd.read_uvh5(filename)
        sim2 = Simulator(data=uvd)

    # Make sure that the data agree to numerical precision.
    assert np.allclose(
        base_sim.data.data_array, sim2.data.data_array, rtol=0, atol=1e-7
    )


def test_io_bad_format(base_sim, tmp_path):
    with pytest.raises(ValueError) as err:
        base_sim.write(tmp_path / "data.bad_extension", save_format="bad_type")
    assert "must correspond to a write method" in err.value.args[0]


@pytest.mark.parametrize("pol", [None, "xx"])
def test_get_full_data(ref_sim, pol):
    data = ref_sim.get("noiselike_eor", pol=pol)
    if pol is None:
        assert np.allclose(data, ref_sim.data.data_array, rtol=0, atol=1e-7)
    else:
        assert np.allclose(data, ref_sim.data.data_array[..., 0], rtol=0, atol=1e-7)


@pytest.mark.parametrize("pol", [None, "xx"])
def test_get_with_one_seed(base_sim, pol):
    # Set the seed mode to "once" even if that's not realistic.
    base_sim.add("noiselike_eor", seed="once")
    ant1, ant2 = (1, 0)  # Use convention ant1 > ant2 for conjugation considerations.
    data = base_sim.get("noiselike_eor", ant1=ant1, ant2=ant2, pol=pol)
    antpairpol = (ant1, ant2) if pol is None else (ant1, ant2, pol)
    true_data = base_sim.data.get_data(antpairpol)
    if pol is None:
        assert np.allclose(data[..., 0], true_data, rtol=0, atol=1e-7)
    else:
        assert np.allclose(data, true_data, rtol=0, atol=1e-7)


def test_get_nonexistent_component(ref_sim):
    with pytest.raises(AttributeError) as err:
        ref_sim.get("diffuse_foreground")
    assert "has not been simulated" in err.value.args[0]


def test_get_without_specifying_antenna(ref_sim):
    with pytest.raises(TypeError) as err:
        ref_sim.get("noiselike_eor", ant1=1, ant2=None)
    assert "specify an antenna pair" in err.value.args[0]


def test_not_add_vis(base_sim):
    vis = base_sim.add("noiselike_eor", add_vis=False, ret_vis=True)

    assert np.all(base_sim.data.data_array == 0)

    assert not np.all(vis == 0)

    assert "noiselike_eor" not in base_sim.data.history
    assert "noiselike_eor" not in base_sim._components.keys()

    # make sure None is returned if neither adding nor returning
    assert base_sim.add("noiselike_eor", add_vis=False, ret_vis=False) is None


def test_adding_vis_but_also_returning(base_sim):
    vis = base_sim.add("noiselike_eor", ret_vis=True)

    assert not np.all(vis == 0)
    assert np.all(np.isclose(vis, base_sim.data.data_array))

    # use season defaults for simplicity
    defaults.set("h1c")
    vis += base_sim.add("diffuse_foreground", ret_vis=True)
    # deactivate defaults for good measure
    defaults.deactivate()
    assert np.all(np.isclose(vis, base_sim.data.data_array))


def test_filter():
    sim = create_sim(autos=True)

    # only add visibilities for the (0,1) baseline
    vis_filter = (0, 1, "xx")

    sim.add("noiselike_eor", vis_filter=vis_filter)
    assert np.all(sim.data.get_data(0, 0) == 0)
    assert np.all(sim.data.get_data(1, 1) == 0)
    assert np.all(sim.data.get_data(0, 1) != 0)
    assert np.all(sim.data.get_data(1, 0) != 0)
    assert np.all(sim.data.get_data(0, 1) == sim.data.get_data(1, 0).conj())


def test_consistent_across_reds():
    # initialize a sim with some redundant baselines
    # this is a 7-element hex array
    ants = hex_array(2, split_core=False, outriggers=0)
    sim = Simulator(
        Nfreqs=20,
        start_freq=1e8,
        channel_width=5e6,
        Ntimes=20,
        start_time=2458115.9,
        integration_time=10.7,
        array_layout=ants,
    )

    # activate season defaults for simplicity
    defaults.set("h1c")

    # add something that should be the same across a redundant group
    sim.add("diffuse_foreground", seed="redundant")

    # deactivate defaults for good measure
    defaults.deactivate()

    reds = sim._get_reds()[1]  # choose non-autos
    # check that every pair in the redundant group agrees
    for i, _bl1 in enumerate(reds):
        for bl2 in reds[i + 1 :]:
            # get the antenna pairs from the baseline integers
            bl1 = sim.data.baseline_to_antnums(_bl1)
            bl2 = sim.data.baseline_to_antnums(bl2)
            vis1 = sim.data.get_data(bl1)
            vis2 = sim.data.get_data(bl2)
            assert np.all(np.isclose(vis1, vis2))

    # Check that seeds vary between redundant groups
    seeds = list(list(sim._seeds.values())[0].values())
    assert all(
        seed_pair[0] != seed_pair[1] for seed_pair in itertools.combinations(seeds, 2)
    )


def test_run_sim():
    # activate season defaults for simplicity
    defaults.set("h1c", refresh=True)

    sim_params = {
        "diffuse_foreground": {"Tsky_mdl": HERA_Tsky_mdl["xx"]},
        "pntsrc_foreground": {"nsrcs": 500, "Smin": 0.1},
        "noiselike_eor": {"eor_amp": 3e-2},
        "thermal_noise": {"Tsky_mdl": HERA_Tsky_mdl["xx"], "integration_time": 8.59},
        "rfi_scatter": {
            "scatter_chance": 0.99,
            "scatter_strength": 5.7,
            "scatter_std": 2.2,
        },
        "rfi_impulse": {"impulse_chance": 0.99, "impulse_strength": 17.22},
        "rfi_stations": {},
        "gains": {"gain_spread": 0.05},
        "sigchain_reflections": {
            "amp": [0.5, 0.5],
            "dly": [14, 7],
            "phs": [0.7723, 3.2243],
        },
        "whitenoise_xtalk": {"amplitude": 1.2345},
    }

    sim = create_sim(autos=True)

    sim.run_sim(**sim_params)

    assert not np.all(np.isclose(sim.data.data_array, 0))

    # instantiate a mock simulation file
    tmp_sim_file = tempfile.mkstemp()[1]
    # write something to it
    with open(tmp_sim_file, "w") as sim_file:
        sim_file.write(
            """
            diffuse_foreground:
                Tsky_mdl: !Tsky
                    datafile: {}/HERA_Tsky_Reformatted.npz
                    pol: yy
            pntsrc_foreground:
                nsrcs: 500
                Smin: 0.1
            noiselike_eor:
                eor_amp: 0.03
            gains:
                gain_spread: 0.05
            cross_coupling_xtalk:
                amp: 0.225
                dly: 13.2
                phs: 2.1123
            thermal_noise:
                Tsky_mdl: !Tsky
                    datafile: {}/HERA_Tsky_Reformatted.npz
                    pol: xx
                integration_time: 9.72
            rfi_scatter:
                scatter_chance: 0.99
                scatter_strength: 5.7
                scatter_std: 2.2
                """.format(
                DATA_PATH, DATA_PATH
            )
        )
    sim = create_sim(autos=True)
    sim.run_sim(tmp_sim_file)
    assert not np.all(np.isclose(sim.data.data_array, 0))

    # deactivate season defaults for good measure
    defaults.deactivate()


def test_run_sim_both_args(base_sim, tmp_path):
    # make a temporary test file
    tmp_sim_file = tmp_path / "temp_config.yaml"
    with open(tmp_sim_file, "w") as sim_file:
        sim_file.write(
            """
            pntsrc_foreground:
                nsrcs: 5000
                """
        )
    sim_params = {"diffuse_foreground": {"Tsky_mdl": HERA_Tsky_mdl["xx"]}}
    with pytest.raises(ValueError) as err:
        base_sim.run_sim(tmp_sim_file, **sim_params)
    assert "Please only pass one of the two." in err.value.args[0]


def test_bad_yaml_config(base_sim, tmp_path):
    # make a bad config file
    tmp_sim_file = tmp_path / "bad_config.yaml"
    with open(tmp_sim_file, "w") as sim_file:
        sim_file.write(
            """
            this:
                is: a
                 bad: file
                 """
        )
    with pytest.raises(IOError) as err:
        base_sim.run_sim(tmp_sim_file)
    assert err.value.args[0] == "The configuration file was not able to be loaded."


def test_run_sim_bad_param_key(base_sim):
    bad_key = {"something": {"something else": "another different thing"}}
    with pytest.raises(UnboundLocalError) as err:
        base_sim.run_sim(**bad_key)
    assert "The component 'something' wasn't found." in err.value.args[0]


def test_run_sim_bad_param_value(base_sim):
    bad_value = {"diffuse_foreground": 13}
    with pytest.raises(TypeError) as err:
        base_sim.run_sim(**bad_value)
    assert "The parameters for diffuse_foreground are not" in err.value.args[0]
