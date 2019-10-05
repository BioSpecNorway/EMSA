import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from src.emsc import emsc


def test_multiplicative_correction():
    n_channels = 100

    base_spectrum = np.random.uniform(0, 2, n_channels)
    spectra = np.array([
        base_spectrum * 1,
        base_spectrum * 3,
        base_spectrum * 5,
    ])
    wavenumbers = np.array(range(n_channels))
    corrected_spectra = emsc(spectra, wavenumbers)

    assert_array_almost_equal(corrected_spectra[0], base_spectrum * 3)
    assert_array_almost_equal(corrected_spectra[1], base_spectrum * 3)
    assert_array_almost_equal(corrected_spectra[2], base_spectrum * 3)


def test_reference_spectrum():
    n_channels = 100

    base_spectrum = np.random.uniform(0, 2, n_channels)
    spectra = np.array([
        base_spectrum * 1,
        base_spectrum * 3,
        base_spectrum * 5,
    ])
    wavenumbers = np.array(range(n_channels))

    corrected_spectra = emsc(spectra, wavenumbers, reference=base_spectrum)

    assert_array_almost_equal(corrected_spectra[0], base_spectrum)
    assert_array_almost_equal(corrected_spectra[1], base_spectrum)
    assert_array_almost_equal(corrected_spectra[2], base_spectrum)


def test_linear_correction():
    n_channels = 100

    base_spectrum = np.random.uniform(0, 2, n_channels)
    linear_coef = -0.6135

    spectra = np.array([
        base_spectrum * 1,
        base_spectrum * 3,
        base_spectrum * 5 + np.linspace(-1, 1, 100)*linear_coef,
    ])
    wavenumbers = np.array(range(n_channels))
    corrected_spectra, coefs = emsc(spectra, wavenumbers,
                                    reference=base_spectrum,
                                    return_coefs=True)

    assert_array_almost_equal(corrected_spectra[0], base_spectrum)
    assert_array_almost_equal(corrected_spectra[1], base_spectrum)
    assert_array_almost_equal(corrected_spectra[2], base_spectrum)

    # check linear coefs
    assert_almost_equal(coefs[0, 2], 0)
    assert_almost_equal(coefs[1, 2], 0)
    assert_almost_equal(coefs[2, 2], linear_coef)


def test_constituents():
    n_channels = 100

    base_spectrum = np.random.uniform(0, 2, n_channels)
    constituent = np.random.uniform(0, 2, n_channels)

    spectra = np.array([
        base_spectrum,
        base_spectrum * 3 + constituent * 2 + np.linspace(-1, 1, n_channels) * 4
    ])
    constituents = np.array([
        constituent
    ])
    wavenumbers = np.array(range(n_channels))

    corrected_spectra, coefs = emsc(spectra, wavenumbers,
                                    reference=base_spectrum,
                                    constituents=constituents,
                                    return_coefs=True)

    assert_array_almost_equal(corrected_spectra[0], base_spectrum)
    assert_array_almost_equal(corrected_spectra[1], base_spectrum)

    # check coefs
    assert_almost_equal(coefs[1, 0], 3)
    assert_almost_equal(coefs[1, 1], 2)
    assert_almost_equal(coefs[1, 2], 0)
    assert_almost_equal(coefs[1, 3], 4)
    assert_almost_equal(coefs[1, 4], 0)
