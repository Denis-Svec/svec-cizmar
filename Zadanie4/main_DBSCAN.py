import numpy as np
import cv2 as cv
import open3d as o3d
import matplotlib.pyplot as plt


def display_outlier(cld, indx):
    inlier_points = cld.select_by_index(indx)
    outlier_cloud = cld.select_by_index(indx, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_points.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_points, outlier_cloud])



pcd = o3d.io.read_point_cloud('output.pcd')
#out = np.asarray(pcd.points)

# visualize
#o3d.visualization.draw_geometries([pcd])
#pcd = o3d.io.read_point_cloud('TLS_kitchen.ply')
# pcd = o3d.io.read_point_cloud('cow_and_lady_gt.ply')
#o3d.visualization.draw_geometries([pcd2])

pcd = pcd.voxel_down_sample(voxel_size=0.02)
cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.08)

display_outlier(pcd, ind)
pcd = pcd.select_by_index(ind)
o3d.visualization.draw_geometries([pcd])
plane_model, inliers = pcd.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=1000)


seg_parts={}
segm_surf={}
max=10
temp_pcd=pcd

for i in range(max):
    colors = plt.get_cmap("tab20")(i)
    seg_parts[i], inliers = temp_pcd.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=1000)
    segm_surf[i]=temp_pcd.select_by_index(inliers)
    labels = np.array(segm_surf[i].cluster_dbscan(eps=0.1 , min_points=10))
    candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)]
    best_candidate=int(np.unique(labels)[np.where(candidates==np.max(candidates))[0]])
    print("the best candidate is: ", best_candidate)
    temp_pcd = temp_pcd.select_by_index(inliers, invert=True) + segm_surf[i].select_by_index(list(np.where(labels != best_candidate)[0]))
    segm_surf[i]=segm_surf[i].select_by_index(list(np.where(labels == best_candidate)[0]))
    segm_surf[i].paint_uniform_color(list(colors[:3]))
    print("pass", i + 1,"/", max, "done.")

o3d.visualization.draw_geometries([segm_surf[i] for i in range(max)] + [temp_pcd])

labels = np.array(temp_pcd.cluster_dbscan(eps=0.05, min_points=5))
max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
temp_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# o3d.visualization.draw_geometries([segments.values()])
o3d.visualization.draw_geometries([segm_surf[i] for i in range(max)] + [temp_pcd])
# o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest],zoom=0.3199,front=[0.30159062875123849, 0.94077325609922868, 0.15488309545553303],lookat=[-3.9559999108314514, -0.055000066757202148, -0.27599999308586121],up=[-0.044411423633999815, -0.138726419067636, 0.98753122516983349])
# o3d.visualization.draw_geometries([rest])