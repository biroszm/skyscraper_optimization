import numpy as np
from tower import create_tower


def point_in_polygon(point, polygon):
    """
    Ray casting algorithm for point-in-polygon.
    Returns True if point is inside or on boundary.
    """
    x, y = point
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Check if point is exactly on a segment
        if point_on_segment(point, (x1, y1), (x2, y2)):
            return True

        intersects = ((y1 > y) != (y2 > y))
        if intersects:
            x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_intersect:
                inside = not inside

    return inside


def point_on_segment(point, a, b, eps=1e-9):
    """
    Check if point lies on line segment ab.
    """
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


def is_tower_inside_parcel(parcel, tower):
    """
    Returns True only if every tower corner is inside or on boundary of parcel.
    """
    footprint = tower["footprint"]
    return all(point_in_polygon(corner, parcel) for corner in footprint)


def random_candidate(parcel, width=60, depth=60, height=200, rng=None, max_tries=10000):
    """
    Generate one valid random tower candidate fully inside the parcel.

    Returns:
        dict with cx, cy, angle_deg, tower
    """
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
                "tower": tower
            }

    raise RuntimeError("Could not generate a valid tower inside the parcel.")


def create_initial_population(
    parcel,
    population_size=400,
    width=60,
    depth=60,
    height=200,
    seed=42
):
    """
    Create an initial GA population of valid tower placements.
    Each individual contains position, angle, and tower geometry.
    """
    rng = np.random.default_rng(seed)
    population = []

    while len(population) < population_size:
        individual = random_candidate(
            parcel=parcel,
            width=width,
            depth=depth,
            height=height,
            rng=rng
        )
        population.append(individual)

    return population