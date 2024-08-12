import matplotlib.pyplot as plt
import numpy as np

# Create a 3D plot with grid lines on the plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create data points that lie on a 2D plane within a 3D space
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = 0.5 * X + 0.2 * Y  # Plane equation: z = 0.5x + 0.2y

# Plot the plane
ax.plot_surface(X, Y, Z, alpha=0.5, rstride=100, cstride=100)

# Plot some points on the plane
x_points = np.random.uniform(-5, 5, 100)
y_points = np.random.uniform(-5, 5, 100)
z_points = 0.5 * x_points + 0.2 * y_points
ax.scatter(x_points, y_points, z_points, color='r')

# Add grid lines on the plane
for i in range(-5, 6):
    ax.plot([i, i], [-5, 5], [0.5*i + 0.2*(-5), 0.5*i + 0.2*5], color='gray', linestyle='--')
    ax.plot([-5, 5], [i, i], [0.5*(-5) + 0.2*i, 0.5*5 + 0.2*i], color='gray', linestyle='--')

# Label axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Data Points in 3D Space Distributed on a 2D Plane with Grid Lines')

plt.savefig('plot-dm-new.png')
