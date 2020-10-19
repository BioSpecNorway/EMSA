import os

import numpy as np
import matplotlib.pyplot as plt

from src.emsa import EMSA
from src.emsc import emsc


PROJECT_DIR = os.path.dirname(__file__) + '/..'


def augment_one_spectrum(spectrum, wavenumbers, label, emsa):
    generator = emsa.generator(
        [spectrum], [label],  # need to be passed array of spectra
        batch_size=5)

    n_times = 3
    for i, batch in enumerate(generator):
        if i >= n_times:
            break
        plt.title('Augmented spectra from one spectrum')
        plt.plot(wavenumbers, spectrum, label='original')
        for augmented_spectrum, label in zip(*batch):
            plt.plot(wavenumbers, augmented_spectrum, label='augmented')
        plt.gca().invert_xaxis()
        plt.legend()
        plt.show()


def augment_dataset(spectra, wavenumbers, labels, emsa):
    generator = emsa.generator(
        spectra, labels,
        equalize_subsampling=True, shuffle=True,
        batch_size=6
    )

    n_times = 3
    for i, batch in enumerate(generator):
        if i >= n_times:
            break
        plt.title('Augmented spectra from dataset')
        for augmented_spectrum, label in zip(*batch):
            plt.plot(wavenumbers, augmented_spectrum, label=label)
        plt.gca().invert_xaxis()
        plt.legend()
        plt.show()


def main(spectra: np.ndarray, wavenumbers: np.ndarray, labels: np.ndarray):
    reference = spectra.mean(axis=0)
    _, coefs_ = emsc(
        spectra, wavenumbers,
        order=2, reference=reference,
        return_coefs=True)
    coefs_std = coefs_.std(axis=0)

    emsa = EMSA(coefs_std, wavenumbers, reference, order=2)

    spectrum_idx = 0
    augment_one_spectrum(
        spectra[spectrum_idx], wavenumbers,
        labels[spectrum_idx], emsa)

    augment_dataset(spectra, wavenumbers, labels, emsa)


if __name__ == '__main__':
    dataset_name = 'corns'
    dataset_path = os.path.join(PROJECT_DIR, 'datasets', dataset_name)
    wns_ = np.load(os.path.join(dataset_path, 'wavenumbers.npy')).astype(np.float)
    spectra_ = np.load(os.path.join(dataset_path, f'{dataset_name}_spectra.npy'))
    markup_ = np.load(os.path.join(dataset_path, f'{dataset_name}_markup.npy'))
    labels_ = markup_[:, 4]

    main(spectra_, wns_, labels_)
