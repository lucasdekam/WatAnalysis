"""
Functions related to water dynamics
"""

import numpy as np


def calc_vector_autocorrelation(
    max_tau: int,
    delta_tau: int,
    step: int,
    vectors: np.ndarray,
    mask: np.ndarray,
):
    """
    Calculate the autocorrelation function for a vector quantity over time.

    Parameters
    ----------
    max_tau : int
        Maximum lag time to calculate ACF for
    delta_tau : int
        Time interval between lag times (points on the C(tau) vs. tau curve)
    step : int
        Step size for time origins. If equal to max_tau, there is no overlap between
        time windows considered in the calculation (so more uncorrelated).
    vectors : numpy.ndarray
        Array of vectors with shape (num_timesteps, num_particles, 3)
    mask : numpy.ndarray
        Boolean mask array indicating which particles to include, shape
        (num_timesteps, num_particles)

    Returns
    -------
    tau : numpy.ndarray
        Array of lag times
    acf : numpy.ndarray
        Normalized autocorrelation function values for each lag time
    """
    tau = np.arange(start=0, stop=max_tau, step=delta_tau)
    acf = np.zeros(tau.shape)
    mask = np.expand_dims(mask, axis=2)

    # Calculate ACF for each lag time
    for i, t in enumerate(tau):
        n_selected_vectors = None
        if t == 0:
            # For t=0, just calculate the dot product with itself
            dot_products = np.sum(
                vectors * vectors * mask, axis=2
            )  # Shape: (num_timesteps, num_molecules)
            n_selected_vectors = np.sum(mask)
        else:
            # For t > 0, calculate the dot products between shifted arrays
            _vectors_0 = vectors[:-t:step] * mask[:-t:step]  # dipole(t=0)
            _vectors_t = vectors[t::step] * mask[t::step]  # dipole(t=tau)
            dot_products = np.sum(
                _vectors_0 * _vectors_t, axis=2
            )  # Shape: ((num_timesteps - t)//step, num_molecules)
            n_selected_vectors = np.sum(mask[:-t:step] * mask[t::step])

        # Average over molecules and time origins
        acf[i] = np.sum(dot_products) / n_selected_vectors

    # Normalize the ACF
    acf /= acf[0]  # Normalize by the zero-lag value
    return tau, acf


def calc_survival_probability(
    max_tau: int,
    delta_tau: int,
    step: int,
    mask: np.ndarray,
):
    """
    Calculate the survival probability.

    Parameters
    ----------
    max_tau : int
        Maximum lag time to calculate ACF for
    delta_tau : int
        Time interval between lag times (points on the C(tau) vs. tau curve)
    step : int
        Step size for time origins. If equal to max_tau, there is no overlap between
        time windows considered in the calculation (so more uncorrelated).
    mask : numpy.ndarray
        Boolean mask array indicating which molecules are in the region of interest for
        all time steps, shape (num_timesteps, num_molecules)

    Returns
    -------
    tau : numpy.ndarray
        Array of lag times
    acf : numpy.ndarray
        Survival probability values for each lag time
    """
    tau_range = np.arange(start=0, stop=max_tau, step=delta_tau)
    acf = np.zeros(tau_range.shape)

    # Calculate continuous ACF for each lag time
    for i, tau in enumerate(tau_range):

        if tau > 0:
            # N(t), shape: (num_timesteps - tau, )
            n_t = np.sum(mask, axis=1)[:-tau:step]

            # shape: ((num_timesteps - tau)//step, num_molecules)
            intersection = np.ones(mask[:-tau:step].shape)
            for k in range(tau):
                intersection *= mask[k : -tau + k : step]
            intersection *= mask[tau::step]

            # N(t,tau), shape: (num_timesteps - tau, )
            n_t_tau = np.sum(intersection, axis=1)

            acf[i] = np.mean(n_t_tau / n_t)
        else:
            acf[i] = 1

    # Normalize the ACF
    acf /= acf[0]  # Normalize by the zero-lag value
    return tau_range, acf
