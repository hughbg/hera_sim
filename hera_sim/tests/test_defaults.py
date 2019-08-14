from os.path import join
import numpy as np
from hera_sim.defaults import defaults
from hera_sim.config import CONFIG_PATH
from hera_sim.sigchain import gen_bandpass
from hera_sim.noise import bm_poly_to_omega_p

if defaults._version_is_compatible:
    def test_config_swap():
        defaults.set_defaults('h1c')
        config1 = defaults._config
        defaults.set_defaults('h2c')
        assert config1 != defaults._config

    def test_direct_config_path():
        config = join(CONFIG_PATH, 'HERA_H2C_CONFIG.yaml')
        defaults.set_defaults(config)

    def test_beam_poly_changes():
        defaults.set_defaults('h1c')
        defaults.activate_defaults()
        fqs = np.linspace(0.1,0.2,100)
        omega_p = bm_poly_to_omega_p(fqs)
        defaults.set_defaults('h2c')
        assert not np.all(omega_p==bm_poly_to_omega_p(fqs))

    def test_bandpass_changes():
        defaults.set_defaults('h1c')
        defaults.activate_defaults()
        fqs = np.linspace(0.1,0.2,100)
        np.random.seed(0)
        bp = gen_bandpass(fqs, [0])[0]
        defaults.set_defaults('h2c')
        np.random.seed(0)
        assert not np.all(bp==gen_bandpass(fqs,[0])[0])
        defaults.deactivate_defaults()

    def test_activate_and_deactivate():
        defaults.activate_defaults()
        assert defaults._use_season_defaults
        defaults.deactivate_defaults()
        assert not defaults._use_season_defaults
