import os
import re
import json
import open3d as o3d
from utils import remove_box, SuscapeScene
import numpy as np
import argparse
import copy

parser = argparse.ArgumentParser(description='adjust ego pose')
parser.add_argument('data', type=str, help="")
parser.add_argument('--lidar', type=str, default="", help="")
parser.add_argument('--scenes', type=str, default=".*", help="")
parser.add_argument('--save-color', type=bool, default=False, help="")
args = parser.parse_args()


def remove_objects(pts, objs):
    
    # remove egocar head & tail
    filter = (pts[:,1] > 4) | (pts[:,1] < -4) | (pts[:,0] > 2) | (pts[:,0] < -2)

    for obj in objs:
        filter = filter & remove_box(pts[:, :3], obj, 0, 1.1) 
    
    return pts[filter]


def combine_and_save_lidars(lidars, poses, file):
     
    map = []
    for i, lidar in enumerate(lidars):
         pts = np.matmul(np.concatenate([lidar[:, 0:3], np.ones([lidar.shape[0],1])], axis=1), poses[i].T)
         l = np.concatenate([pts[:, 0:3], lidar[:, 3:]], axis=1)
         map.append(l)
    
    map = np.concatenate(map, axis=0)
    map = map.astype(np.float32)

    if args.save_color:
        color = (map[:, 4:7]*256.0).astype(np.uint8).astype(np.int32)

        color = (color[:,0] * 0x100  + color[:,1])*0x100 + color[:,2]
        color = color.astype(np.int32)

    size = map.shape[0]
    with open(file, 'wb') as f:
        if args.save_color:
            header = f"""# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z intensity rgb
SIZE 4 4 4 4 4
TYPE F F F F I
COUNT 1 1 1 1 1
WIDTH {size}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {size}
DATA binary
"""
        else:
            header = f"""# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {size}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {size}
DATA binary
"""            
            
            
        f.write(header.encode('utf-8'))
        for i,d in enumerate(map[:, :4]): 
            f.write(d.tobytes())
            if args.save_color:
                f.write(color[i].tobytes())
                
        

def proc_scene(scene):
    scene = SuscapeScene(args.data, scene, args.lidar)
    frames = scene.meta['frames']

    lidars = []
    poses = []

    for i in range(len(frames)):

        next_frame = frames[i]
        next_lidar = scene.read_lidar(next_frame)

        objs = scene.get_boxes_by_frame(next_frame)
        next_lidar = remove_objects(next_lidar, objs)
        lidars.append(next_lidar)

        pose = scene.read_lidar_pose(next_frame)
        pose = np.array(pose['lidarPose']).reshape(4,4)
        poses.append(pose)
                    
    
    os.makedirs(os.path.join(args.data, scene.name, 'map'), exist_ok=True)

    combine_and_save_lidars(lidars, poses, os.path.join(args.data, scene.name, 'map', 'map.pcd'))




scenes = os.listdir(args.data)
scenes.sort()
for s in scenes:
    if re.fullmatch(args.scenes, s):
        print('processing', s)
        proc_scene(s)