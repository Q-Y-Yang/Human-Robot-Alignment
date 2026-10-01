import numpy as np
# import matplotlib.pyplot as plt

directions = [
    {"translation_direction_deg": 90,       "weight": 0.1},
    {"translation_direction_deg": 270,      "weight": 0.1},
    {"translation_direction_deg": 0,        "weight": 1.1},
    {"translation_direction_deg": 45,       "weight": 0.15},
    {"translation_direction_deg": -45,      "weight": 0.15},
    {"translation_direction_deg": 45 / 2,   "weight": 0.10},
    {"translation_direction_deg": -45 / 2,  "weight": 0.10},
    {"translation_direction_deg": -135 / 2, "weight": 0.15},
    {"translation_direction_deg": 135 / 2,  "weight": 0.15},
]

resolution = 0.05
r_inner = 0.55
r_outer = 3.60

# Create grid coordinates
x = np.arange(-r_outer, r_outer + resolution / 2, resolution)
y = np.arange(0.0, r_outer + resolution / 2, resolution)

X, Y = np.meshgrid(x, y)

# A semi-annulus-shaped grid
# R = np.sqrt(X**2 + Y**2)

# mask = (R >= r_inner) & (R <= r_outer)

# X_valid = X[mask]
# Y_valid = Y[mask]

# # Visualize meshgrid
# plt.figure(figsize=(12, 6))

# plt.scatter(
#     X_valid.ravel(),
#     Y_valid.ravel(),
#     s=2
# )

# plt.xlabel("X [m]")
# plt.ylabel("Y [m]")
# plt.title(f"Meshgrid, resolution = {resolution} m")

# plt.axis("equal")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# print("X shape:", X.shape)
# print("Y shape:", Y.shape)

def create_weighted_gaussian_map(
    X,
    Y,
    directions,
    distance,
    sigma=0.1
):


    gaussian_map = np.zeros_like(X, dtype=float)

    center_points = []

    for item in directions:
        angle_deg = item["translation_direction_deg"]
        weight = item["weight"]

        angle_rad = np.deg2rad(angle_deg)


        center_x = - distance * np.sin(angle_rad)
        center_y = distance * np.cos(angle_rad)

        # 2D Gaussian
        gaussian = np.exp(
            -(
                (X - center_x) ** 2
                + (Y - center_y) ** 2
            )
            / (2.0 * sigma**2)
        )
        # print("gaussian max:", np.nanmax(gaussian))
        # multiply with weights
        gaussian_map +=  weight * gaussian
        # print("gaussian map max after wighting:", np.nanmax(weight * gaussian))

        center_points.append(
            {
                "angle_deg": angle_deg,
                "x": center_x,
                "y": center_y,
                "weight": weight,
            }
        )

    return X, Y, gaussian_map, center_points

_, _, gaussian_map, center_points = create_weighted_gaussian_map(
      X=X,
      Y=Y,
      directions=directions,
      distance=0.9,
      sigma=0.1,
  )

# gaussian_map = np.where(mask, gaussian_map, np.nan)

# X, Y, gaussian_map, center_points = (
#     create_weighted_gaussian_map(
#         directions=directions,
#         distance=2.0,
#         sigma=0.25,
#         # resolution=0.05,
#         # x_range=(-3.0, 3.0),
#         # y_range=(-3.0, 3.0),
#         X=X_valid,
#         Y=Y_valid
#     )
# )
# density = np.load("large_interaction_kde_density.npy")
# print("density max:", density.max())
# # print("gaussian map density:", np.nanmax(gaussian_map))
# gaussian_map += density
# print(gaussian_map.size)
# Visualization
# plt.figure(figsize=(8, 7))

# plt.contourf(
#     X,
#     Y,
#     gaussian_map,
#     levels=50,
#     cmap="viridis",
# )

# plt.colorbar(label="Weighted Gaussian value")

# # 标出各高斯中心
# for point in center_points:
#     plt.scatter(
#         point["x"],
#         point["y"],
#         color="red",
#         s=25,
#     )

#     plt.text(
#         point["x"],
#         point["y"],
#         f'{point["angle_deg"]:.1f}°\nw={point["weight"]}',
#         fontsize=8,
#     )

# plt.scatter(0, 0, color="white", edgecolor="black", label="Origin")

# max_index = np.unravel_index(np.nanargmax(gaussian_map), gaussian_map.shape)
# print("max value:", np.nanmax(gaussian_map))
# max_x, max_y = X[max_index], Y[max_index]
# plt.scatter(
#     max_x,
#     max_y,
#     marker="*",
#     color="cyan",
#     edgecolor="black",
#     s=180,
#     zorder=5,
#     label="Maximum",
# )
# plt.annotate(
#     f"({max_x:.2f}, {max_y:.2f})",
#     (max_x, max_y),
#     xytext=(10, 10),
#     textcoords="offset points",
#     color="cyan",
#     zorder=6,
# )

# plt.xlabel("X")
# plt.ylabel("Y")
# plt.axis("equal")
# plt.legend()
# plt.tight_layout()
# plt.show()