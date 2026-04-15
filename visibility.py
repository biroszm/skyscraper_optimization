import numpy as np


def generate_windows_on_tower(tower, window_size=2.0):
    """
    Divide the 4 side faces of the tower into window patches of size window_size x window_size.

    Returns:
        windows: list of dicts, each with:
            - face_id
            - center: np.array([x, y, z])
            - normal: np.array([nx, ny, nz])
            - u_dir: direction along the face edge
            - v_dir: vertical direction
            - width
            - height
    """
    footprint = tower["footprint"]
    height = tower["height"]

    windows = []
    n = len(footprint)

    # footprint assumed CCW
    for i in range(n):
        p0 = footprint[i]
        p1 = footprint[(i + 1) % n]

        edge_vec = p1 - p0
        edge_length = np.linalg.norm(edge_vec)

        if edge_length == 0:
            continue

        u_dir = edge_vec / edge_length
        # outward normal for CCW polygon
        normal_xy = np.array([edge_vec[1], -edge_vec[0]])
        normal_xy = normal_xy / np.linalg.norm(normal_xy)

        normal = np.array([normal_xy[0], normal_xy[1], 0.0])
        v_dir = np.array([0.0, 0.0, 1.0])

        n_u = int(edge_length // window_size)
        n_v = int(height // window_size)

        for iu in range(n_u):
            for iv in range(n_v):
                u = (iu + 0.5) * window_size
                z = (iv + 0.5) * window_size

                center_xy = p0 + u * u_dir
                center = np.array([center_xy[0], center_xy[1], z], dtype=float)

                windows.append({
                    "face_id": i,
                    "center": center,
                    "normal": normal.copy(),
                    "u_dir": np.array([u_dir[0], u_dir[1], 0.0]),
                    "v_dir": v_dir.copy(),
                    "width": window_size,
                    "height": window_size,
                })

    return windows


def ray_box_intersection(origin, direction, box_min, box_max):
    """
    Ray / axis-aligned box intersection using slab method.

    Args:
        origin: np.array([x, y, z])
        direction: np.array([dx, dy, dz]) assumed normalized or nonzero
        box_min: np.array([xmin, ymin, zmin])
        box_max: np.array([xmax, ymax, zmax])

    Returns:
        distance to first intersection along ray, or None if no hit
    """
    tmin = -np.inf
    tmax = np.inf
    eps = 1e-12

    for k in range(3):
        if abs(direction[k]) < eps:
            # Ray parallel to slab
            if origin[k] < box_min[k] or origin[k] > box_max[k]:
                return None
        else:
            t1 = (box_min[k] - origin[k]) / direction[k]
            t2 = (box_max[k] - origin[k]) / direction[k]

            t_near = min(t1, t2)
            t_far = max(t1, t2)

            tmin = max(tmin, t_near)
            tmax = min(tmax, t_far)

            if tmin > tmax:
                return None

    if tmax < 0:
        return None

    # first forward hit
    if tmin >= 0:
        return tmin
    elif tmax >= 0:
        return tmax

    return None


def building_to_box(building):
    """
    Convert building dict to axis-aligned 3D box.
    """
    box_min = np.array([building["x"], building["y"], 0.0], dtype=float)
    box_max = np.array([
        building["x"] + building["width"],
        building["y"] + building["depth"],
        building["height"]
    ], dtype=float)
    return box_min, box_max


def compute_window_intersections(windows, buildings, offset=1e-3):
    """
    For each window, cast a ray along the outward normal and find nearest building hit.

    Args:
        windows: output of generate_windows_on_tower(...)
        buildings: list of building dicts
        offset: small shift along normal so ray starts just outside the tower face

    Returns:
        results: list of dicts with:
            - window_id
            - face_id
            - center
            - normal
            - hit: bool
            - hit_building_id: int or None
            - distance: float or np.inf
            - intersection_point: np.array([x,y,z]) or None
    """
    results = []

    boxes = [building_to_box(b) for b in buildings]

    for idx, w in enumerate(windows):
        origin = w["center"] + offset * w["normal"]
        direction = w["normal"]

        best_dist = np.inf
        best_building_id = None
        best_point = None

        for b_id, (box_min, box_max) in enumerate(boxes):
            dist = ray_box_intersection(origin, direction, box_min, box_max)

            if dist is not None and dist < best_dist:
                best_dist = dist
                best_building_id = b_id
                best_point = origin + dist * direction

        results.append({
            "window_id": idx,
            "face_id": w["face_id"],
            "center": w["center"],
            "normal": w["normal"],
            "hit": best_building_id is not None,
            "hit_building_id": best_building_id,
            "distance": best_dist if best_building_id is not None else np.inf,
            "intersection_point": best_point
        })

    return results