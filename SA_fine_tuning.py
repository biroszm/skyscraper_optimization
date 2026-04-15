import csv
from itertools import product

from site_layout import create_environment, plot_environment
from simulated_annealing_search import (
    run_simulated_annealing,
    plot_convergence,
    plot_temperature,
)
from visibility import generate_windows_on_tower, compute_window_intersections
from scoring import compute_tower_fitness

# Parameter grid
initial_temperature_values = [0.05, 0.1, 0.2]
cooling_rate_values = [0.95, 0.97, 0.99]
step_xy_values = [2, 5, 8]
step_angle_values = [10, 20, 30]

# Create environment once
parcel, buildings = create_environment()

# Store results
all_results = []

for initial_temperature, cooling_rate, step_xy, step_angle in product(
    initial_temperature_values,
    cooling_rate_values,
    step_xy_values,
    step_angle_values,
):
    print(
        f"\nRunning config: "
        f"initial_temperature={initial_temperature}, "
        f"cooling_rate={cooling_rate}, "
        f"step_xy={step_xy}, "
        f"step_angle={step_angle}"
    )

    # Fast search with coarse windows
    best_individual, history = run_simulated_annealing(
        parcel=parcel,
        buildings=buildings,
        width=40,
        depth=40,
        height=300,
        window_size=10.0,          # coarse for speed
        dmax=150.0,
        scale=40.0,
        no_hit_bonus=1.2,
        seed=42,
        max_iterations=300,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        step_xy=float(step_xy),
        step_angle=float(step_angle),
        stop_window=10,
        min_relative_improvement=0.01
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
    print(f"Coarse SA fitness: {best_individual['fitness']:.6f}")
    print(f"Precise fitness:   {final_fitness_precise:.6f}")
    print(f"cx:    {best_individual['cx']:.2f}")
    print(f"cy:    {best_individual['cy']:.2f}")
    print(f"angle: {best_individual['angle_deg']:.2f}")

    all_results.append({
        "initial_temperature": initial_temperature,
        "cooling_rate": cooling_rate,
        "step_xy": step_xy,
        "step_angle": step_angle,
        "coarse_sa_fitness": best_individual["fitness"],
        "final_fitness_precise": final_fitness_precise,
        "cx": best_individual["cx"],
        "cy": best_individual["cy"],
        "angle_deg": best_individual["angle_deg"],
    })

# Save results to CSV
csv_filename = "simulated_annealing_parameter_sweep_results.csv"
fieldnames = [
    "initial_temperature",
    "cooling_rate",
    "step_xy",
    "step_angle",
    "coarse_sa_fitness",
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