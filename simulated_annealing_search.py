import numpy as np
import matplotlib.pyplot as plt

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
            return {
                "cx": cx,
                "cy": cy,
                "angle_deg": angle_deg,
                "tower": tower
            }

    raise RuntimeError("Could not sample a valid feasible candidate.")


def propose_neighbor(
    current,
    parcel,
    width,
    depth,
    height,
    rng,
    step_xy=5.0,
    step_angle=20.0,
    max_tries=2000
):
    """
    Propose a nearby valid solution by perturbing cx, cy, angle.
    """
    for _ in range(max_tries):
        new_cx = current["cx"] + rng.normal(0.0, step_xy)
        new_cy = current["cy"] + rng.normal(0.0, step_xy)
        new_angle = (current["angle_deg"] + rng.normal(0.0, step_angle)) % 360.0

        tower = create_tower(
            cx=new_cx,
            cy=new_cy,
            width=width,
            depth=depth,
            height=height,
            angle_deg=new_angle
        )

        if is_tower_inside_parcel(parcel, tower):
            return {
                "cx": new_cx,
                "cy": new_cy,
                "angle_deg": new_angle,
                "tower": tower
            }

    # fallback: keep current if no valid neighbor found
    return {
        "cx": current["cx"],
        "cy": current["cy"],
        "angle_deg": current["angle_deg"],
        "tower": current["tower"]
    }


def should_stop(best_history, window=10, min_relative_improvement=0.05):
    if len(best_history) < window + 1:
        return False

    old = best_history[-(window + 1)]
    new = best_history[-1]

    if abs(old) < 1e-12:
        return False

    relative_improvement = (new - old) / abs(old)
    return relative_improvement < min_relative_improvement


def run_simulated_annealing(
    parcel,
    buildings,
    width=60,
    depth=60,
    height=200,
    window_size=8.0,
    dmax=150.0,
    scale=40.0,
    no_hit_bonus=1.2,
    seed=42,
    max_iterations=300,
    initial_temperature=0.1,
    cooling_rate=0.97,
    step_xy=5.0,
    step_angle=20.0,
    stop_window=10,
    min_relative_improvement=0.05
):
    rng = np.random.default_rng(seed)

    # initial feasible solution
    current = sample_random_feasible_candidate(
        parcel=parcel,
        width=width,
        depth=depth,
        height=height,
        rng=rng
    )
    current_fitness = evaluate_tower_fitness(
        current["tower"],
        buildings,
        window_size=window_size,
        dmax=dmax,
        scale=scale,
        no_hit_bonus=no_hit_bonus
    )
    current["fitness"] = current_fitness

    best = {
        "cx": current["cx"],
        "cy": current["cy"],
        "angle_deg": current["angle_deg"],
        "tower": current["tower"],
        "fitness": current["fitness"]
    }

    best_history = [best["fitness"]]
    current_history = [current["fitness"]]
    temperature_history = [initial_temperature]

    T = initial_temperature

    for iteration in range(max_iterations):
        candidate = propose_neighbor(
            current=current,
            parcel=parcel,
            width=width,
            depth=depth,
            height=height,
            rng=rng,
            step_xy=step_xy,
            step_angle=step_angle
        )

        candidate_fitness = evaluate_tower_fitness(
            candidate["tower"],
            buildings,
            window_size=window_size,
            dmax=dmax,
            scale=scale,
            no_hit_bonus=no_hit_bonus
        )
        candidate["fitness"] = candidate_fitness

        delta = candidate_fitness - current["fitness"]

        # accept if better, or sometimes if worse
        if delta >= 0:
            accept = True
        else:
            accept_prob = np.exp(delta / max(T, 1e-12))
            accept = rng.uniform() < accept_prob

        if accept:
            current = candidate

        if current["fitness"] > best["fitness"]:
            best = {
                "cx": current["cx"],
                "cy": current["cy"],
                "angle_deg": current["angle_deg"],
                "tower": current["tower"],
                "fitness": current["fitness"]
            }

        best_history.append(best["fitness"])
        current_history.append(current["fitness"])
        temperature_history.append(T)

        print(
            f"Iter {iteration:3d} | "
            f"T = {T:.5f} | "
            f"current = {current['fitness']:.6f} | "
            f"best = {best['fitness']:.6f} | "
            f"cx = {current['cx']:.2f}, "
            f"cy = {current['cy']:.2f}, "
            f"angle = {current['angle_deg']:.2f}"
        )

        if should_stop(
            best_history,
            window=stop_window,
            min_relative_improvement=min_relative_improvement
        ):
            print("\nStopping criterion reached.")
            break

        T *= cooling_rate

    history = {
        "best_fitness": best_history,
        "current_fitness": current_history,
        "temperature": temperature_history
    }

    return best, history


def plot_convergence(history):
    iterations = np.arange(len(history["best_fitness"]))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, history["best_fitness"], label="Best fitness")
    plt.plot(iterations, history["current_fitness"], label="Current fitness")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Simulated Annealing Convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_temperature(history):
    iterations = np.arange(len(history["temperature"]))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, history["temperature"])
    plt.xlabel("Iteration")
    plt.ylabel("Temperature")
    plt.title("Simulated Annealing Temperature Schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.show()