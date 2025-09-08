import matplotlib.pyplot as plt
import numpy as np

def plot_track(x_values: list[float], y_values: list[float], headings: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", "box")

    # 1) Turn off the big offset so you see full metre ticks
    ax.ticklabel_format(style="plain", useOffset=False, axis="both")

    # 2) Draw the backbone of the route as a solid black line
    ax.plot(x_values, y_values, "-", color="black", linewidth=2, label="Path")

    # 3) Plot arrows with quiver (vectorized, so you can scale them up)
    U = np.cos(headings)
    V = np.sin(headings)
    norm = plt.Normalize(0, len(x_values) - 1)
    cmap = plt.cm.plasma

    # Only plot every Nth arrow so it doesn’t get too crowded
    N = max(1, len(x_values) // 100)
    ax.quiver(
        x_values[::N],
        y_values[::N],
        U[::N],
        V[::N],
        color=cmap(norm(np.arange(len(x_values))[::N])),
        scale=20,         # adjust to lengthen/shorten arrows
        width=0.005,
        label="Heading"
    )

    # 4) Colorbar just for the heading colormap
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.1, fraction=0.05)
    cbar.set_label("Waypoint index")

    # 5) Labels, legend, grid
    ax.set_title("Track waypoints")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc="upper left")
    ax.grid(True)

    plt.show()
