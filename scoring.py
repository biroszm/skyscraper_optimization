import numpy as np


def score_window(result, dmax=150.0, scale=40.0, no_hit_bonus=1.2):
    """
    Score a single window based on the nearest building hit.

    Rules:
    - no hit -> bonus score
    - hit -> score increases with distance and saturates

    Args:
        result: one dict from compute_window_intersections(...)
        dmax: cap for useful view distance
        scale: controls how quickly score rises with distance
        no_hit_bonus: score assigned if no building is hit

    Returns:
        float
    """
    if not result["hit"]:
        return no_hit_bonus

    d = min(result["distance"], dmax)
    return 1.0 - np.exp(-d / scale)


def score_all_windows(results, dmax=150.0, scale=40.0, no_hit_bonus=1.2):
    """
    Add a score field to each window result.

    Returns:
        scored_results: list of dicts
    """
    scored_results = []

    for r in results:
        r_new = r.copy()
        r_new["score"] = score_window(
            r,
            dmax=dmax,
            scale=scale,
            no_hit_bonus=no_hit_bonus
        )
        scored_results.append(r_new)

    return scored_results


def compute_tower_fitness(results, dmax=150.0, scale=40.0, no_hit_bonus=1.2):
    """
    Compute average window score for the tower.

    Returns:
        float
    """
    if len(results) == 0:
        return 0.0

    scores = [
        score_window(r, dmax=dmax, scale=scale, no_hit_bonus=no_hit_bonus)
        for r in results
    ]

    return float(np.mean(scores))