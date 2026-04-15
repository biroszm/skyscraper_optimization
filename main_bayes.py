from time import perf_counter

from site_layout import create_environment, plot_environment
from bayes_opt_search import run_bayesian_optimization, plot_convergence
from visibility import generate_windows_on_tower, compute_window_intersections
from scoring import compute_tower_fitness

parcel, buildings = create_environment()

start_time = perf_counter()

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
    n_initial=30,
    n_iterations=80,
    candidate_pool_size=1000,
    seed=42,
    stop_window=100,
    min_relative_improvement=0.01,
    xi=0.001
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

end_time = perf_counter()
elapsed_time = end_time - start_time

print("\nFinal best solution:")
print(f"Coarse BO fitness: {best_individual['fitness']:.6f}")
print(f"Precise fitness:   {final_fitness_precise:.6f}")
print(f"cx:    {best_individual['cx']:.2f}")
print(f"cy:    {best_individual['cy']:.2f}")
print(f"angle: {best_individual['angle_deg']:.2f}")
print(f"Execution time: {elapsed_time:.4f} seconds")

plot_environment(parcel, buildings, tower=final_tower, mode="both")
plot_convergence(history)