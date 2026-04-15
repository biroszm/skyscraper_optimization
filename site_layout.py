import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def create_environment():
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


def extrude_polygon_3d(footprint, height):
    bottom = [[x, y, 0] for x, y in footprint]
    top = [[x, y, height] for x, y in footprint]

    faces = [bottom, top]
    n = len(footprint)

    for i in range(n):
        j = (i + 1) % n
        side = [
            [footprint[i][0], footprint[i][1], 0],
            [footprint[j][0], footprint[j][1], 0],
            [footprint[j][0], footprint[j][1], height],
            [footprint[i][0], footprint[i][1], height],
        ]
        faces.append(side)

    return faces


def plot_environment_2d(parcel, buildings, tower=None):
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
    parcel_poly = Polygon(
        parcel_closed,
        facecolor="orange",
        edgecolor="red",
        linewidth=2,
        alpha=0.7
    )
    ax.add_patch(parcel_poly)

    # draw tower if provided
    if tower is not None:
        tower_closed = np.vstack([tower["footprint"], tower["footprint"][0]])
        tower_poly = Polygon(
            tower_closed,
            facecolor="steelblue",
            edgecolor="navy",
            linewidth=2,
            alpha=0.85
        )
        ax.add_patch(tower_poly)

        ax.text(
            tower["cx"],
            tower["cy"],
            f'Tower\n{tower["height"]}m',
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_aspect("equal")
    ax.set_xlim(-180, 220)
    ax.set_ylim(-160, 200)
    ax.set_title("Parcel, Buildings, and Tower - 2D")
    ax.grid(False)
    plt.show()


def plot_environment_3d(parcel, buildings, tower=None):
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

    # draw tower if provided
    if tower is not None:
        tower_faces = extrude_polygon_3d(tower["footprint"], tower["height"])
        tower_poly3d = Poly3DCollection(
            tower_faces,
            facecolors="steelblue",
            edgecolors="navy",
            alpha=0.8
        )
        ax.add_collection3d(tower_poly3d)

    ax.set_xlim(-180, 220)
    ax.set_ylim(-160, 200)
    ax.set_zlim(0, 300)
    ax.set_title("Parcel, Buildings, and Tower - 3D")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Height (m)")
    ax.view_init(elev=25, azim=45)
    ax.grid(False)
    plt.show()


def plot_environment(parcel, buildings, tower=None, mode="2d"):
    if mode == "2d":
        plot_environment_2d(parcel, buildings, tower=tower)
    elif mode == "3d":
        plot_environment_3d(parcel, buildings, tower=tower)
    elif mode == "both":
        plot_environment_2d(parcel, buildings, tower=tower)
        plot_environment_3d(parcel, buildings, tower=tower)
    else:
        raise ValueError("mode must be '2d', '3d', or 'both'")