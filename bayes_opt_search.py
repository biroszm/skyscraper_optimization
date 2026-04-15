import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from tower import create_tower
from visibility import generate_windows_on_tower, compute_window_intersections
from scoring import compute_tower_fitness


def point_on_segment(point, a, b, eps=1e-9):
    px, py = point
    ax, ay = a
    bx, by = b

    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False

    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    if dot < -eps:
        return False

    sq_len = (bx - ax) ** 2 + (by - ay) ** 2
    if dot - sq_len > eps:
        return False

    return True


def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if point_on_segment(point, (x1, y1), (x2, y2)):
            return True

        intersects = ((y1 > y) != (y2 > y))
        if intersects:
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside

    return inside


def is_tower_inside_parcel(parcel, tower):
    return all(point_in_polygon(corner, parcel) for corner in tower["footprint"])


def evaluate_tower_fitness(
    tower,
    buildings,
    window_size=8.0,
    dmax=150.0,
    scale=40.0,
    no_hit_bonus=1.2
):
    windows = generate_windows_on_tower(tower, window_size=window_size)
    results = compute_window_intersections(windows, buildings)

    fitness = compute_tower_fitness(
        results,
        dmax=dmax,
        scale=scale,
        no_hit_bonus=no_hit_bonus
    )
    return fitness


def encode_candidate(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    return np.array([cx, cy, np.sin(angle_rad), np.cos(angle_rad)], dtype=float)


def sample_random_feasible_candidate(parcel, width, depth, height, rng, max_tries=20000):
    xmin = np.min(parcel[:, 0])
    xmax = np.max(parcel[:, 0])
    ymin = np.min(parcel[:, 1])
    ymax = np.max(parcel[:, 1])

    for _ in range(max_tries):
        cx = rng.uniform(xmin, xmax)
        cy = rng.uniform(ymin, ymax)
        angle_deg = rng.uniform(0.0, 360.0)

        tower = create_tower(
            cx=cx,
            cy=cy,
            width=width,
            depth=depth,
            height=height,
            angle_deg=angle_deg
        )

        if is_tower_inside_parcel(parcel, tower):
            return cx, cy, angle_deg, tower

    raise RuntimeError("Could not sample a valid feasible tower candidate.")


def sample_feasible_candidate_pool(parcel, width, depth, height, n_candidates, rng):
    candidates = []

    while len(candidates) < n_candidates:
        cx, cy, angle_deg, tower = sample_random_feasible_candidate(
            parcel=parcel,
            width=width,
            depth=depth,
            height=height,
            rng=rng
        )
        candidates.append({
            "cx": cx,
            "cy": cy,
            "angle_deg": angle_deg,
            "tower": tower
        })

    return candidates


def expected_improvement(mu, sigma, best_y, xi=0.01):
    """
    EI for maximization.
    """
    sigma = np.maximum(sigma, 1e-12)
    improvement = mu - best_y - xi
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-12] = 0.0
    return ei


def should_stop(best_history, window=10, min_relative_improvement=0.05):
    if len(best_history) < window + 1:
        return False

    old = best_history[-(window + 1)]
    new = best_history[-1]

    if abs(old) < 1e-12:
        return False

    relative_improvement = (new - old) / abs(old)
    return relative_improvement < min_relative_improvement


def run_bayesian_optimization(
    parcel,
    buildings,
    width=60,
    depth=60,
    height=200,
    window_size=8.0,
    dmax=150.0,
    scale=40.0,
    no_hit_bonus=1.2,
    n_initial=20,
    n_iterations=80,
    candidate_pool_size=2000,
    seed=42,
    stop_window=10,
    min_relative_improvement=0.05,
    xi=0.01
):
    """
    Bayesian optimization over (cx, cy, angle).

    Steps:
    1. Sample feasible initial points
    2. Evaluate them
    3. Fit GP surrogate on [cx, cy, sin(angle), cos(angle)]
    4. Sample a feasible candidate pool
    5. Choose next point with max Expected Improvement
    6. Repeat until stop
    """
    rng = np.random.default_rng(seed)

    X = []
    y = []
    evaluated = []
    best_history = []
    mean_history = []

    best_individual = None

    # --------------------------------
    # Initial design
    # --------------------------------
    for i in range(n_initial):
        cx, cy, angle_deg, tower = sample_random_feasible_candidate(
            parcel=parcel,
            width=width,
            depth=depth,
            height=height,
            rng=rng
        )

        fitness = evaluate_tower_fitness(
            tower=tower,
            buildings=buildings,
            window_size=window_size,
            dmax=dmax,
            scale=scale,
            no_hit_bonus=no_hit_bonus
        )

        X.append(encode_candidate(cx, cy, angle_deg))
        y.append(fitness)

        individual = {
            "cx": cx,
            "cy": cy,
            "angle_deg": angle_deg,
            "tower": tower,
            "fitness": fitness
        }
        evaluated.append(individual)

        if best_individual is None or fitness > best_individual["fitness"]:
            best_individual = individual

        print(
            f"Initial {i:2d} | fitness = {fitness:.6f} | "
            f"cx = {cx:.2f}, cy = {cy:.2f}, angle = {angle_deg:.2f}"
        )

    best_history.append(max(y))
    mean_history.append(float(np.mean(y)))

    # --------------------------------
    # BO loop
    # --------------------------------
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(4), nu=2.5)
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
    )

    for it in range(n_iterations):
        X_arr = np.array(X, dtype=float)
        y_arr = np.array(y, dtype=float)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=seed
        )
        gp.fit(X_arr, y_arr)

        pool = sample_feasible_candidate_pool(
            parcel=parcel,
            width=width,
            depth=depth,
            height=height,
            n_candidates=candidate_pool_size,
            rng=rng
        )

        X_pool = np.array([
            encode_candidate(c["cx"], c["cy"], c["angle_deg"]) for c in pool
        ], dtype=float)

        mu, sigma = gp.predict(X_pool, return_std=True)
        best_y = np.max(y_arr)
        ei = expected_improvement(mu, sigma, best_y, xi=xi)

        best_idx = int(np.argmax(ei))
        chosen = pool[best_idx]

        fitness = evaluate_tower_fitness(
            tower=chosen["tower"],
            buildings=buildings,
            window_size=window_size,
            dmax=dmax,
            scale=scale,
            no_hit_bonus=no_hit_bonus
        )

        X.append(encode_candidate(chosen["cx"], chosen["cy"], chosen["angle_deg"]))
        y.append(fitness)

        individual = {
            "cx": chosen["cx"],
            "cy": chosen["cy"],
            "angle_deg": chosen["angle_deg"],
            "tower": chosen["tower"],
            "fitness": fitness
        }
        evaluated.append(individual)

        if fitness > best_individual["fitness"]:
            best_individual = individual

        current_best = float(np.max(y))
        current_mean = float(np.mean(y))
        best_history.append(current_best)
        mean_history.append(current_mean)

        print(
            f"BO iter {it:3d} | "
            f"new fitness = {fitness:.6f} | "
            f"best = {current_best:.6f} | "
            f"cx = {chosen['cx']:.2f}, "
            f"cy = {chosen['cy']:.2f}, "
            f"angle = {chosen['angle_deg']:.2f}"
        )

        if should_stop(
            best_history,
            window=stop_window,
            min_relative_improvement=min_relative_improvement
        ):
            print("\nStopping criterion reached.")
            break

    history = {
        "best_fitness": best_history,
        "mean_fitness": mean_history
    }

    return best_individual, history, evaluated


def plot_convergence(history):
    generations = np.arange(len(history["best_fitness"]))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, history["best_fitness"], label="Best fitness")
    plt.plot(generations, history["mean_fitness"], label="Mean fitness")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Bayesian Optimization Convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()