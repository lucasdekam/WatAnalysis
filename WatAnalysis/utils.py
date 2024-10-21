# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np


def get_cum_ave(data):
    cum_sum = data.cumsum()
    cum_ave = cum_sum / (np.arange(len(data)) + 1)
    return cum_ave


def bin_edges_to_grid(bin_edges: np.ndarray):
    return bin_edges[:-1] + np.diff(bin_edges) / 2
