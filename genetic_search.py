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


def random_candidate(parcel, width=60, depth=60, height=200, rng=None, max_tries=20000):
    if rng is None:
        rng = np.random.default_rng()

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
                "tower": tower,
                "fitness": None
            }

    raise RuntimeError("Could not generate a valid random candidate inside the parcel.")


def create_initial_population(
    parcel,
    population_size=400,
    width=60,
    depth=60,
    height=200,
    seed=42
):
    rng = np.random.default_rng(seed)
    population = []

    while len(population) < population_size:
        ind = random_candidate(
            parcel=parcel,
            width=width,
            depth=depth,
            height=height,
            rng=rng
        )
        population.append(ind)

    return population


def evaluate_individual(
    individual,
    buildings,
    window_size=2.0,
    dmax=150.0,
    scale=40.0,
    no_hit_bonus=1.2
):
    tower = individual["tower"]
    windows = generate_windows_on_tower(tower, window_size=window_size)
    results = compute_window_intersections(windows, buildings)

    fitness = compute_tower_fitness(
        results,
        dmax=dmax,
        scale=scale,
        no_hit_bonus=no_hit_bonus
    )

    individual["fitness"] = fitness
    return fitness


def evaluate_population(
    population,
    buildings,
    window_size=2.0,
    dmax=150.0,
    scale=40.0,
    no_hit_bonus=1.2
):
    for ind in population:
        evaluate_individual(
            ind,
            buildings,
            window_size=window_size,
            dmax=dmax,
            scale=scale,
            no_hit_bonus=no_hit_bonus
        )
    return population


def select_best(population, n_best=40):
    population_sorted = sorted(population, key=lambda x: x["fitness"], reverse=True)
    return population_sorted[:n_best]


def mutate_best_population(
    best_population,
    parcel,
    width=60,
    depth=60,
    height=200,
    rng=None,
    n_angle_mutations=4,
    n_position_mutations=5,
    position_step=8.0,
    max_tries_per_child=2000
):
    """
    For each selected parent:
    - keep the parent unchanged
    - create angle-only mutations (same cx, cy; new angle)
    - create position-only mutations (same angle; perturbed cx, cy)

    All children must remain fully inside the parcel.
    """

    if rng is None:
        rng = np.random.default_rng()

    new_population = []

    for parent in best_population:
        # Keep parent
        kept_tower = create_tower(
            cx=parent["cx"],
            cy=parent["cy"],
            width=width,
            depth=depth,
            height=height,
            angle_deg=parent["angle_deg"]
        )

        new_population.append({
            "cx": parent["cx"],
            "cy": parent["cy"],
            "angle_deg": parent["angle_deg"],
            "tower": kept_tower,
            "fitness": None
        })

        # -------------------------------------------------
        # Angle-only mutations: same cx, cy; new angle
        # -------------------------------------------------
        created_angle = 0
        tries = 0

        while created_angle < n_angle_mutations and tries < max_tries_per_child:
            tries += 1
            new_angle = rng.uniform(0.0, 360.0)

            tower = create_tower(
                cx=parent["cx"],
                cy=parent["cy"],
                width=width,
                depth=depth,
                height=height,
                angle_deg=new_angle
            )

            if is_tower_inside_parcel(parcel, tower):
                new_population.append({
                    "cx": parent["cx"],
                    "cy": parent["cy"],
                    "angle_deg": new_angle,
                    "tower": tower,
                    "fitness": None
                })
                created_angle += 1

        if created_angle < n_angle_mutations:
            raise RuntimeError(
                f"Could not generate {n_angle_mutations} valid angle mutations for "
                f"parent at cx={parent['cx']:.2f}, cy={parent['cy']:.2f}"
            )

        # -------------------------------------------------
        # Position-only mutations: same angle; new cx, cy
        # -------------------------------------------------
        created_pos = 0
        tries = 0

        while created_pos < n_position_mutations and tries < max_tries_per_child:
            tries += 1

            new_cx = parent["cx"] + rng.normal(0.0, position_step)
            new_cy = parent["cy"] + rng.normal(0.0, position_step)

            tower = create_tower(
                cx=new_cx,
                cy=new_cy,
                width=width,
                depth=depth,
                height=height,
                angle_deg=parent["angle_deg"]
            )

            if is_tower_inside_parcel(parcel, tower):
                new_population.append({
                    "cx": new_cx,
                    "cy": new_cy,
                    "angle_deg": parent["angle_deg"],
                    "tower": tower,
                    "fitness": None
                })
                created_pos += 1

        if created_pos < n_position_mutations:
            raise RuntimeError(
                f"Could not generate {n_position_mutations} valid position mutations for "
                f"parent at cx={parent['cx']:.2f}, cy={parent['cy']:.2f}"
            )

    return new_population


def should_stop(best_history, window=10, min_relative_improvement=0.05):
    if len(best_history) < window + 1:
        return False

    old = best_history[-(window + 1)]
    new = best_history[-1]

    if old == 0:
        return False

    relative_improvement = (new - old) / abs(old)
    return relative_improvement < min_relative_improvement


def run_ga(
    parcel,
    buildings,
    population_size=400,
    n_best=40,
    width=60,
    depth=60,
    height=200,
    window_size=2.0,
    dmax=150.0,
    scale=40.0,
    no_hit_bonus=1.2,
    seed=42,
    max_generations=200,
    stop_window=10,
    min_relative_improvement=0.05,
    n_angle_mutations=4,
    n_position_mutations=5,
    position_step=8.0
):
    rng = np.random.default_rng(seed)

    expected_population_size = n_best * (1 + n_angle_mutations + n_position_mutations)
    if population_size != expected_population_size:
        raise ValueError(
            f"population_size must equal n_best * (1 + n_angle_mutations + n_position_mutations). "
            f"Got population_size={population_size}, expected {expected_population_size}."
        )

    population = create_initial_population(
        parcel=parcel,
        population_size=population_size,
        width=width,
        depth=depth,
        height=height,
        seed=seed
    )

    best_history = []
    mean_history = []
    best_individual = None

    for generation in range(max_generations):
        population = evaluate_population(
            population,
            buildings,
            window_size=window_size,
            dmax=dmax,
            scale=scale,
            no_hit_bonus=no_hit_bonus
        )

        population_sorted = sorted(population, key=lambda x: x["fitness"], reverse=True)
        current_best = population_sorted[0]
        current_best_fitness = current_best["fitness"]
        current_mean_fitness = float(np.mean([ind["fitness"] for ind in population]))

        best_history.append(current_best_fitness)
        mean_history.append(current_mean_fitness)

        if best_individual is None or current_best_fitness > best_individual["fitness"]:
            best_individual = {
                "cx": current_best["cx"],
                "cy": current_best["cy"],
                "angle_deg": current_best["angle_deg"],
                "tower": current_best["tower"],
                "fitness": current_best["fitness"]
            }

        print(
            f"Generation {generation:3d} | "
            f"best = {current_best_fitness:.6f} | "
            f"mean = {current_mean_fitness:.6f} | "
            f"cx = {current_best['cx']:.2f}, "
            f"cy = {current_best['cy']:.2f}, "
            f"angle = {current_best['angle_deg']:.2f}"
        )

        if should_stop(
            best_history,
            window=stop_window,
            min_relative_improvement=min_relative_improvement
        ):
            print("\nStopping criterion reached.")
            break

        selected = select_best(population, n_best=n_best)

        population = mutate_best_population(
            selected,
            parcel=parcel,
            width=width,
            depth=depth,
            height=height,
            rng=rng,
            n_angle_mutations=n_angle_mutations,
            n_position_mutations=n_position_mutations,
            position_step=position_step
        )

    history = {
        "best_fitness": best_history,
        "mean_fitness": mean_history
    }

    return best_individual, history


def plot_convergence(history):
    generations = np.arange(len(history["best_fitness"]))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, history["best_fitness"], label="Best fitness")
    plt.plot(generations, history["mean_fitness"], label="Mean fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("GA Convergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()