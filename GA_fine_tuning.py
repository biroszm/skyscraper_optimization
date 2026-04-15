import csv
from itertools import product

from site_layout import create_environment, plot_environment
from genetic_search import run_ga, plot_convergence
from visibility import generate_windows_on_tower, compute_window_intersections
from scoring import compute_tower_fitness

# Parameter grid
n_best_values = [20, 40, 60]
min_relative_improvement_values = [0.001, 0.01, 0.5]

# Create environment once
parcel, buildings = create_environment()

# Store all results
all_results = []

for n_best, min_relative_improvement in product(
    n_best_values,
    min_relative_improvement_values
):
    print(
        f"\nRunning config: "
        f"n_best={n_best}, "
        f"min_relative_improvement={min_relative_improvement}"
    )

    best_individual, history = run_ga(
        parcel=parcel,
        buildings=buildings,
        population_size=400,
        n_best=n_best,
        width=40,
        depth=40,
        height=300,
        window_size=10.0,
        dmax=150.0,
        scale=40.0,
        no_hit_bonus=1.2,
        seed=42,
        max_generations=200,
        stop_window=10,
        min_relative_improvement=min_relative_improvement
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

    print("\nFinal best solution:")
    print(f"Coarse GA fitness: {best_individual['fitness']:.6f}")
    print(f"Precise fitness:   {final_fitness_precise:.6f}")
    print(f"cx:    {best_individual['cx']:.2f}")
    print(f"cy:    {best_individual['cy']:.2f}")
    print(f"angle: {best_individual['angle_deg']:.2f}")

    all_results.append({
        "n_best": n_best,
        "min_relative_improvement": min_relative_improvement,
        "coarse_ga_fitness": best_individual["fitness"],
        "final_fitness_precise": final_fitness_precise,
        "cx": best_individual["cx"],
        "cy": best_individual["cy"],
        "angle_deg": best_individual["angle_deg"],
    })

# Save results to CSV
csv_filename = "genetic_algorithm_parameter_sweep_results.csv"
fieldnames = [
    "n_best",
    "min_relative_improvement",
    "coarse_ga_fitness",
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