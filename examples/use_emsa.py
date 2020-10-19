import os

import numpy as np
import matplotlib.pyplot as plt

from src.emsa import EMSA
from src.emsc import emsc


PROJECT_DIR = os.path.dirname(__file__) + '/..'


def plot_from_generator(generator, wavenumbers, n_times=3,
                        spectrum=None, title=None, override_label=None):
    for i, batch in enumerate(generator):
        if i >= n_times:
            break
        if title is not None:
            plt.title(title)
        if spectrum is not None:
            plt.plot(wavenumbers, spectrum, label='original', linestyle='--')
        for augmented_spectrum, label in zip(*batch):
            if override_label is not None:
                label = override_label
            plt.plot(wavenumbers, augmented_spectrum, label=label)
        plt.gca().invert_xaxis()
        plt.legend()
        plt.show()


def augment_one_spectrum(spectrum, wavenumbers, label, emsa):
    generator = emsa.generator([spectrum], [label], batch_size=5)

    plot_from_generator(generator, wavenumbers,
                        title='Augmented spectra from one spectrum',
                        override_label='augmented')


def augment_dataset(spectra, wavenumbers, labels, emsa):
    generator = emsa.generator(
        spectra, labels,
        equalize_subsampling=True, shuffle=True,
        batch_size=6
    )

    plot_from_generator(
        generator, wavenumbers, title='Augmented spectra from dataset')


def main(spectra: np.ndarray, wavenumbers: np.ndarray, labels: np.ndarray):
    reference = spectra.mean(axis=0)
    _, coefs_ = emsc(
        spectra, wavenumbers,
        order=2, reference=reference,
        return_coefs=True)
    coefs_std = coefs_.std(axis=0)

    emsa = EMSA(coefs_std, wavenumbers, reference, order=2)

    # (1) Augment one spectrum
    spectrum_idx = 0
    augment_one_spectrum(
        spectra[spectrum_idx], wavenumbers,
        labels[spectrum_idx], emsa)

    # (2) Augment whole dataset
    augment_dataset(spectra, wavenumbers, labels, emsa)


if __name__ == '__main__':
    dataset_name = 'corns'
    dataset_path = os.path.join(PROJECT_DIR, 'datasets', dataset_name)
    wns_ = np.load(os.path.join(dataset_path, 'wavenumbers.npy')).astype(np.float)
    spectra_ = np.load(os.path.join(dataset_path, f'{dataset_name}_spectra.npy'))
    markup_ = np.load(os.path.join(dataset_path, f'{dataset_name}_markup.npy'))
    labels_ = markup_[:, 4]

    main(spectra_, wns_, labels_)
