import numpy as np


def calculate_iv(agg):
    eps = 1e-6
    total_good = (agg["total"] - agg["bads"]).sum()
    total_bad = agg["bads"].sum()

    dist_good = (agg["total"] - agg["bads"]) / (total_good + eps)
    dist_bad = agg["bads"] / (total_bad + eps)

    woe = np.log((dist_good + eps) / (dist_bad + eps))
    return ((dist_good - dist_bad) * woe).sum()
