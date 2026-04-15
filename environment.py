import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def create_environment():
    """
    Returns:
        parcel: np.ndarray of parcel corner points
        buildings: list of surrounding building dictionaries
    """
    parcel = np.array([
        [0, 0],
        [100, 0],
        [110, 80],
        [-20, 70]
    ], dtype=float)

    buildings = [
        {"x": -120, "y": -40, "width": 40, "depth": 60, "height": 180},
        {"x": -140, "y": 60, "width": 50, "depth": 50, "height": 220},

        {"x": 130, "y": -20, "width": 45, "depth": 60, "height": 200},
        {"x": 160, "y": 70, "width": 40, "depth": 50, "height": 240},

        {"x": 20, "y": 130, "width": 60, "depth": 40, "height": 210},
        {"x": -60, "y": 140, "width": 50, "depth": 50, "height": 190},

        {"x": 10, "y": -120, "width": 60, "depth": 40, "height": 170},
        {"x": -70, "y": -110, "width": 50, "depth": 50, "height": 230},

        {"x": 90, "y": 90, "width": 30, "depth": 30, "height": 260},
        {"x": -90, "y": 20, "width": 35, "depth": 35, "height": 210},
    ]

    return parcel, buildings


def cuboid_faces(x, y, z, dx, dy, dz):
    p0 = [x, y, z]
    p1 = [x + dx, y, z]
    p2 = [x + dx, y + dy, z]
    p3 = [x, y + dy, z]

    p4 = [x, y, z + dz]
    p5 = [x + dx, y, z + dz]
    p6 = [x + dx, y + dy, z + dz]
    p7 = [x, y + dy, z + dz]

    return [
        [p0, p1, p2, p3],
        [p4, p5, p6, p7],
        [p0, p1, p5, p4],
        [p1, p2, p6, p5],
        [p2, p3, p7, p6],
        [p3, p0, p4, p7],
    ]


def plot_environment_2d(parcel, buildings):
    fig, ax = plt.subplots(figsize=(10, 10))

    for b in buildings:
        rect = Rectangle(
            (b["x"], b["y"]),
            b["width"],
            b["depth"],
            facecolor="lightgray",
            edgecolor="black"
        )
        ax.add_patch(rect)

        cx = b["x"] + b["width"] / 2
        cy = b["y"] + b["depth"] / 2
        ax.text(cx, cy, f'{int(b["height"])}m', ha="center", va="center", fontsize=8)

    parcel_closed = np.vstack([parcel, parcel[0]])
    poly = Polygon(parcel_closed, facecolor="orange", edgecolor="red", linewidth=2, alpha=0.7)
    ax.add_patch(poly)

    for i, (x, y) in enumerate(parcel):
        ax.text(x, y, f"P{i+1}", color="red", fontsize=12)

    ax.set_aspect("equal")
    ax.set_title("Parcel and Surrounding Buildings - 2D")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(False)

    ax.set_xlim(-180, 220)
    ax.set_ylim(-160, 200)

    plt.show()


def plot_environment_3d(parcel, buildings):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    for b in buildings:
        faces = cuboid_faces(b["x"], b["y"], 0, b["width"], b["depth"], b["height"])
        poly3d = Poly3DCollection(
            faces,
            facecolors="lightgray",
            edgecolors="black",
            alpha=0.85
        )
        ax.add_collection3d(poly3d)

    parcel_closed = np.vstack([parcel, parcel[0]])
    ax.plot(parcel_closed[:, 0], parcel_closed[:, 1], zs=0, color="red", linewidth=3)

    ax.set_xlim(-180, 220)
    ax.set_ylim(-160, 200)
    ax.set_zlim(0, 300)

    ax.set_title("Parcel and Surrounding Buildings - 3D")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Height (m)")
    ax.view_init(elev=25, azim=45)

    plt.show()


def plot_environment(parcel, buildings, mode="2d"):
    if mode == "2d":
        plot_environment_2d(parcel, buildings)
    elif mode == "3d":
        plot_environment_3d(parcel, buildings)
    elif mode == "both":
        plot_environment_2d(parcel, buildings)
        plot_environment_3d(parcel, buildings)
    else:
        raise ValueError("mode must be '2d', '3d', or 'both'")