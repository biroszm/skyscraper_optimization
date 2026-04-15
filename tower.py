import numpy as np


def get_rotated_rectangle(cx, cy, width, depth, angle_deg):
    angle = np.radians(angle_deg)

    local = np.array([
        [-width / 2, -depth / 2],
        [ width / 2, -depth / 2],
        [ width / 2,  depth / 2],
        [-width / 2,  depth / 2]
    ], dtype=float)

    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    rotated = local @ rotation.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return rotated


def create_tower(cx=40, cy=35, width=60, depth=60, height=200, angle_deg=0):
    footprint = get_rotated_rectangle(cx, cy, width, depth, angle_deg)

    tower = {
        "cx": cx,
        "cy": cy,
        "width": width,
        "depth": depth,
        "height": height,
        "angle_deg": angle_deg,
        "footprint": footprint
    }

    return tower