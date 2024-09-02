# load the pcd file
import open3d as o3d

pcd = o3d.io.read_point_cloud(
    "/home/haoyang/project/haoyang/UniSoft/ThirdParty/AdaptiGraph/src/planning/dump/vis_cloth/cloth_points_50.pcd"
)
o3d.visualization.draw_geometries([pcd])
