import astropy_healpix as aph
import healvis
import numpy as np
import pytest
from astropy.units import sday, rad
from astropy import units
from pyuvsim.analyticbeam import AnalyticBeam
from vis_cpu import HAVE_GPU
from hera_sim.defaults import defaults
from hera_sim import io
from hera_sim.visibilities import VisCPU, HealVis, VisibilitySimulation, ModelData
from pyradiosky import SkyModel
from astropy.coordinates.angles import Latitude, Longitude

SIMULATORS = (HealVis, VisCPU)

if HAVE_GPU:

    class VisGPU(VisCPU):
        """Simple mock class to make testing VisCPU with use_gpu=True easier"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, use_gpu=True, **kwargs)

    SIMULATORS = SIMULATORS + (VisGPU,)


np.random.seed(0)
NTIMES = 10
BM_PIX = 31
NPIX = 12 * 16 ** 2
NFREQ = 5


@pytest.fixture
def uvdata():
    defaults.set("h1c")
    return io.empty_uvdata(
        Nfreqs=NFREQ,
        integration_time=sday.to("s") / NTIMES,
        Ntimes=NTIMES,
        array_layout={
            0: (0, 0, 0),
        },
        start_time=2456658.5,
        conjugation="ant1<ant2",
    )


@pytest.fixture
def uvdataJD():
    defaults.set("h1c")
    return io.empty_uvdata(
        Nfreqs=NFREQ,
        integration_time=sday.to("s") / NTIMES,
        Ntimes=NTIMES,
        array_layout={
            0: (0, 0, 0),
        },
        start_time=2456659,
    )


@pytest.fixture
def sky_model(uvdata):
    return make_point_sky(
        uvdata,
        ra=np.array([0.0]) * rad,
        dec=np.array(uvdata.telescope_location_lat_lon_alt[0]) * rad,
        align=False,
    )


@pytest.fixture
def sky_modelJD(uvdataJD):
    return make_point_sky(
        uvdataJD,
        ra=np.array([0.0]) * rad,
        dec=np.array(uvdata.telescope_location_lat_lon_alt[0]) * rad,
        align=False,
    )


def test_healvis_beam(uvdata, sky_model):
    sim = VisibilitySimulation(
        simulator=HealVis(),
        data_model=ModelData(
            uvdata=uvdata,
            sky_model=sky_model,
        ),
        n_side=2 ** 4,
    )

    assert len(sim.data_model.beams) == 1
    assert isinstance(sim.data_model.beams[0], healvis.beam_model.AnalyticBeam)


def test_healvis_beam_obsparams(tmpdir):
    # Now try creating with an obsparam file
    direc = tmpdir.mkdir("test_healvis_beam")

    with open(direc.join("catalog.txt"), "w") as fl:
        fl.write(
            """SOURCE_ID       RA_J2000 [deg]  Dec_J2000 [deg] Flux [Jy]       Frequency [Hz]
    HERATEST0       68.48535        -28.559917      1       100000000.0
    """
        )

    with open(direc.join("telescope_config.yml"), "w") as fl:
        fl.write(
            """
    beam_paths:
        0 : 'uniform'
    telescope_location: (-30.72152777777791, 21.428305555555557, 1073.0000000093132)
    telescope_name: MWA
    """
        )

    with open(direc.join("layout.csv"), "w") as fl:
        fl.write(
            """Name     Number   BeamID   E          N          U

    Tile061        40        0   -34.8010   -41.7365     1.5010
    Tile062        41        0   -28.0500   -28.7545     1.5060
    Tile063        42        0   -11.3650   -29.5795     1.5160
    Tile064        43        0    -9.0610   -20.7885     1.5160
    """
        )

    with open(direc.join("obsparams.yml"), "w") as fl:
        fl.write(
            """
    freq:
      Nfreqs: 1
      channel_width: 80000.0
      start_freq: 100000000.0
    sources:
      catalog: {0}/catalog.txt
    telescope:
      array_layout: {0}/layout.csv
      telescope_config_name: {0}/telescope_config.yml
    time:
      Ntimes: 1
      integration_time: 11.0
      start_time: 2458098.38824015
    """.format(
                direc.strpath
            )
        )

    sim = VisibilitySimulation(
        data_model=ModelData.from_config(direc.join("obsparams.yml").strpath),
        simulator=HealVis(),
    )
    beam = sim.data_model.beams[0]
    assert isinstance(beam, healvis.beam_model.AnalyticBeam)


def test_JD(uvdata, uvdataJD, sky_model):
    model_data = ModelData(sky_model=sky_model, uvdata=uvdata)

    vis = VisCPU()

    sim1 = VisibilitySimulation(data_model=model_data, simulator=vis).simulate()

    model_data2 = ModelData(sky_model=sky_model, uvdata=uvdataJD)

    sim2 = VisibilitySimulation(data_model=model_data2, simulator=vis).simulate()

    assert sim1.shape == sim2.shape
    assert not np.allclose(sim1, sim2, atol=0.1)


@pytest.fixture
def uvdata2():
    defaults.set("h1c")
    return io.empty_uvdata(
        Nfreqs=NFREQ,
        integration_time=sday.to("s") / NTIMES,
        Ntimes=NTIMES,
        array_layout={0: (0, 0, 0), 1: (1, 1, 0)},
        start_time=2456658.5,
        conjugation="ant1<ant2",
    )


def make_point_sky(uvdata, ra: np.ndarray, dec: np.ndarray, align=True):
    freqs = np.unique(uvdata.freq_array)

    # put a point source in
    point_source_flux = np.ones((len(ra), len(freqs)))

    # align to healpix center for direct comparision
    if align:
        ra, dec = align_src_to_healpix(ra * rad, dec * rad)

    return SkyModel(
        ra=Longitude(ra),
        dec=Latitude(dec),
        stokes=np.array(
            [
                point_source_flux.T,
                np.zeros((len(freqs), len(ra))),
                np.zeros((len(freqs), len(ra))),
                np.zeros((len(freqs), len(ra))),
            ]
        ),
        name=["derp"] * len(ra),
        spectral_type="full",
        freq_array=freqs,
    )


def zenith_sky_model(uvdata2):
    return make_point_sky(
        uvdata2,
        ra=np.array([0.0]),
        dec=np.array([uvdata2.telescope_location_lat_lon_alt[0]]),
        align=True,
    )


def horizon_sky_model(uvdata2):
    return make_point_sky(
        uvdata2,
        ra=np.array([0.0]),
        dec=np.array([uvdata2.telescope_location_lat_lon_alt[0] + np.pi / 2]),
        align=True,
    )


def twin_sky_model(uvdata2):
    return make_point_sky(
        uvdata2,
        ra=np.array([0.0, 0.0]),
        dec=np.array(
            [
                uvdata2.telescope_location_lat_lon_alt[0] + np.pi / 4,
                uvdata2.telescope_location_lat_lon_alt[0],
            ]
        ),
        align=True,
    )


def half_sky_model(uvdata2):
    nbase = 4
    nside = 2 ** nbase

    sky = create_uniform_sky(
        np.unique(uvdata2.freq_array),
        nbase=nbase,
    )

    # Zero out values within pi/2 of (theta=pi/2, phi=0)
    hp = aph.HEALPix(nside=nside, order="ring")
    ipix_disc = hp.cone_search_lonlat(0 * rad, np.pi / 2 * rad, radius=np.pi / 2 * rad)
    sky.stokes[0, :, ipix_disc] = 0
    print(sky.stokes.unit)
    return sky


def create_uniform_sky(freq, nbase=4, scale=1) -> SkyModel:
    """Create a uniform sky with total (integrated) flux density of `scale`"""
    nfreq = len(freq)
    nside = 2 ** nbase
    npix = 12 * nside ** 2
    return SkyModel(
        nside=nside,
        hpx_inds=np.arange(npix),
        stokes=np.array(
            [
                np.ones((nfreq, npix)) * scale / (4 * np.pi),
                np.zeros((nfreq, npix)),
                np.zeros((nfreq, npix)),
                np.zeros((nfreq, npix)),
            ]
        )
        * units.Jy
        / units.sr,
        spectral_type="full",
        freq_array=freq,
    )


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_shapes(uvdata, simulator):
    sky = create_uniform_sky(np.unique(uvdata.freq_array))

    sim = VisibilitySimulation(
        data_model=ModelData(uvdata=uvdata, sky_model=sky),
        simulator=simulator(),
        n_side=2 ** 4,
    )

    assert sim.simulate().shape == (uvdata.Nblts, 1, NFREQ, 1)


@pytest.mark.parametrize("precision, cdtype", [(1, np.complex64), (2, complex)])
def test_dtypes(uvdata, precision, cdtype):
    sky = create_uniform_sky(np.unique(uvdata.freq_array))
    vis = VisCPU(precision=precision)

    sim = VisibilitySimulation(
        data_model=ModelData(uvdata=uvdata, sky_model=sky), simulator=vis
    )

    v = sim.simulate()
    assert v.dtype == cdtype


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_zero_sky(uvdata, simulator):
    sky = create_uniform_sky(np.unique(uvdata.freq_array), scale=0)

    sim = VisibilitySimulation(
        data_model=ModelData(uvdata=uvdata, sky_model=sky), simulator=simulator()
    )
    v = sim.simulate()
    np.testing.assert_equal(v, 0)


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_autocorr_flat_beam(uvdata, simulator):
    sky = create_uniform_sky(np.unique(uvdata.freq_array), nbase=6)

    sim = VisibilitySimulation(
        data_model=ModelData(uvdata=uvdata, sky_model=sky), simulator=simulator()
    )
    v = sim.simulate()

    # Account for factor of 2 between Stokes I and 'xx' pol for vis_cpu
    if simulator == VisCPU:
        v *= 2.0

    np.testing.assert_allclose(np.abs(v), np.mean(v), rtol=1e-5)
    np.testing.assert_almost_equal(np.abs(v), 0.5, 2)


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_single_source_autocorr(uvdata, simulator, sky_model):
    sim = VisibilitySimulation(
        data_model=ModelData(
            uvdata=uvdata,
            sky_model=sky_model,
        ),
        simulator=simulator(),
        n_side=2 ** 4,
    )
    v = sim.simulate()

    # Account for factor of 2 between Stokes I and 'xx' pol for vis_cpu
    if simulator == VisCPU:
        v *= 2.0

    # Make sure the source is over the horizon half the time
    # (+/- 1 because of the discreteness of the times)
    # 1e-3 on either side to account for float inaccuracies.
    assert (
        -1e-3 + (NTIMES / 2.0 - 1.0) / NTIMES
        <= np.round(np.abs(np.mean(v)), 3)
        <= (NTIMES / 2.0 + 1.0) / NTIMES + 1e-3
    )


@pytest.mark.parametrize("simulator", SIMULATORS)
def test_single_source_autocorr_past_horizon(uvdata, simulator):
    sky_model = make_point_sky(
        uvdata,
        ra=np.array([0]) * rad,
        dec=np.array(uvdata.telescope_location_lat_lon_alt[0] + 1.1 * np.pi / 2) * rad,
        align=False,
    )

    sim = VisibilitySimulation(
        data_model=ModelData(
            uvdata=uvdata,
            sky_model=sky_model,
        ),
        simulator=simulator(),
        n_side=2 ** 4,
    )
    v = sim.simulate()

    assert np.abs(np.mean(v)) == 0


def test_viscpu_coordinate_correction(uvdata2, zenith_sky_model):
    sim = VisibilitySimulation(
        data_model=ModelData(
            uvdata=uvdata,
            sky_model=zenith_sky_model,
        ),
        simulator=VisCPU(
            correct_source_positions=True, ref_time="2018-08-31T04:02:30.11"
        ),
    )

    # Apply correction
    # viscpu.correct_point_source_pos(obstime="2018-08-31T04:02:30.11", frame="icrs")
    v = sim.simulate()
    assert np.all(~np.isnan(v))


def align_src_to_healpix(ra, dec, nside=2 ** 4):
    """Where the point sources will be placed when converted to healpix model

    Parameters
    ----------
    point_source_pos : ndarray
        Positions of point sources to be passed to a Simulator.
    point_source_flux : ndarray
        Corresponding fluxes of point sources at each frequency.
    nside : int
        Healpix nside parameter.


    Returns
    -------
    new_pos: ndarray
        Point sources positioned at their nearest healpix centers.
    new_flux: ndarray
        Corresponding new flux values.
    """
    # Get which pixel every point source lies in.
    pix = aph.lonlat_to_healpix(ra, dec, nside)
    ra, dec = aph.healpix_to_lonlat(pix, nside)
    return ra, dec


@pytest.mark.parametrize(
    "sky_model, beam_model",
    [
        (zenith_sky_model, None),
        (horizon_sky_model, None),
        (twin_sky_model, None),
        (half_sky_model, None),
        (half_sky_model, [AnalyticBeam("airy", diameter=1.75)]),
    ],
)
def test_comparison(uvdata2, sky_model, beam_model):
    cpu = VisCPU()
    healvis = HealVis()

    model_data = ModelData(
        uvdata=uvdata2, sky_model=sky_model(uvdata2), beams=beam_model
    )

    viscpu = VisibilitySimulation(data_model=model_data, simulator=cpu).simulate()
    viscpu *= 2.0  # account for factor of 2 between Stokes I and 'xx' pol.

    healvis = VisibilitySimulation(
        data_model=model_data, simulator=healvis, n_side=2 ** 4
    ).simulate()

    assert viscpu.shape == healvis.shape
    np.testing.assert_allclose(viscpu, healvis, rtol=0.05)
