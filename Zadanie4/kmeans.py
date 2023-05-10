import numpy as np
import open3d as o3d


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


pcd = o3d.io.read_point_cloud('output.pcd')

pcd = pcd.voxel_down_sample(voxel_size=0.02)
cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.08)

display_inlier_outlier(pcd, ind)

pcd = pcd.select_by_index(ind)

# Convert point cloud to numpy array
points = np.asarray(pcd.points)

# Cluster the points using K-means algorithm
K = 3  # Number of clusters to create
max_iterations = 500  # Maximum number of iterations for K-means algorithm
labels = np.zeros(len(points))
centers = points[np.random.choice(len(points), K, replace=False)]

for i in range(max_iterations):
    # Assign each point to its closest cluster
    distances = np.linalg.norm(points[:, np.newaxis, :] - centers, axis=2)
    new_labels = np.argmin(distances, axis=1)

    # Check if the labels have changed
    if np.all(new_labels == labels):
        break

    # Update the centers of each cluster
    for j in range(K):
        centers[j] = np.mean(points[new_labels == j], axis=0)

    labels = new_labels

# Visualize the clusters using Open3D
colors = np.random.uniform(size=(K, 3))
cluster_clouds = [o3d.geometry.PointCloud(pcd.select_by_index(np.where(labels == i)[0])) for i in range(K)]
for i, cloud in enumerate(cluster_clouds):
    cloud.paint_uniform_color(colors[i])
o3d.visualization.draw_geometries(cluster_clouds)