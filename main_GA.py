from time import perf_counter

from site_layout import create_environment, plot_environment
from genetic_search import run_ga, plot_convergence

parcel, buildings = create_environment()

start_time = perf_counter()

best_individual, history = run_ga(
    parcel=parcel,
    buildings=buildings,
    population_size=400,
    n_best=40,
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
    min_relative_improvement=0.05
)

end_time = perf_counter()
elapsed_time = end_time - start_time

print("\nFinal best solution:")
print(f"Fitness: {best_individual['fitness']:.6f}")
print(f"cx: {best_individual['cx']:.2f}")
print(f"cy: {best_individual['cy']:.2f}")
print(f"angle: {best_individual['angle_deg']:.2f}")
print(f"Execution time: {elapsed_time:.4f} seconds")

plot_environment(parcel, buildings, tower=best_individual["tower"], mode="both")
plot_convergence(history)