import csv
from itertools import product

from site_layout import create_environment, plot_environment
from bayes_opt_search import run_bayesian_optimization, plot_convergence
from visibility import generate_windows_on_tower, compute_window_intersections
from scoring import compute_tower_fitness

# Parameter grid
n_initial_values = [10, 20, 30]
candidate_pool_sizes = [1000, 2000, 3000]
xi_values = [0.001, 0.01, 0.05]

# Create environment once
parcel, buildings = create_environment()

# Store all results here
all_results = []

for n_initial, candidate_pool_size, xi in product(
    n_initial_values, candidate_pool_sizes, xi_values
):
    print(
        f"\nRunning config: "
        f"n_initial={n_initial}, "
        f"candidate_pool_size={candidate_pool_size}, "
        f"xi={xi}"
    )

    # Fast search with coarse window grid
    best_individual, history, evaluated = run_bayesian_optimization(
        parcel=parcel,
        buildings=buildings,
        width=40,
        depth=40,
        height=300,
        window_size=10.0,          # coarse for speed
        dmax=150.0,
        scale=40.0,
        no_hit_bonus=1.2,
        n_initial=n_initial,
        n_iterations=80,
        candidate_pool_size=candidate_pool_size,
        seed=42,
        stop_window=10,
        min_relative_improvement=0.01,
        xi=xi
    )

    # Final precise evaluation with 2x2 windows
    final_tower = best_individual["tower"]
    windows_fine = generate_windows_on_tower(final_tower, window_size=2.0)
    results_fine = compute_window_intersections(windows_fine, buildings)
    final_fitness_precise = compute_tower_fitness(
        results_fine,
        dmax=150.0,
        scale=40.0,
        no_hit_bonus=1.2
    )

    print("Final best solution:")
    print(f"Coarse BO fitness: {best_individual['fitness']:.6f}")
    print(f"Precise fitness:   {final_fitness_precise:.6f}")
    print(f"cx:    {best_individual['cx']:.2f}")
    print(f"cy:    {best_individual['cy']:.2f}")
    print(f"angle: {best_individual['angle_deg']:.2f}")

    all_results.append({
        "n_initial": n_initial,
        "candidate_pool_size": candidate_pool_size,
        "xi": xi,
        "coarse_bo_fitness": best_individual["fitness"],
        "final_fitness_precise": final_fitness_precise,
        "cx": best_individual["cx"],
        "cy": best_individual["cy"],
        "angle_deg": best_individual["angle_deg"],
    })

# Save results to CSV
csv_filename = "bayes_opt_parameter_sweep_results.csv"
fieldnames = [
    "n_initial",
    "candidate_pool_size",
    "xi",
    "coarse_bo_fitness",
    "final_fitness_precise",
    "cx",
    "cy",
    "angle_deg",
]

with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)

print(f"\nSaved results to {csv_filename}")