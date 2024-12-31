# this file computes statistics of the datasets

from genericpath import isfile
import os
import json
import numpy as np
import re
import argparse
import time
import suscape.dataset as dataset
parser = argparse.ArgumentParser(description='scene stat')        
parser.add_argument('--data', type=str,default='/home/lie/nas/suscape_scenes', help="")
parser.add_argument('--scenes', type=str,default='.*', help="")
args = parser.parse_args()


susc=dataset.SuscapeDataset(args.data)

def stat_one_scene(scene):

    frames = susc.get_frames(scene)
    objs=0
    for f in frames:
        l = susc.read_label(scene, f)
        objs += len(l['objs'])
    

    desc = susc.read_desc(scene)['scene'].replace(',', ';')
    ego_pose = susc.read_ego_pose(scene, frames[0])

    tm = time.gmtime(int(frames[0].split('.')[0]))
    tmstr = time.strftime("%Y-%m-%d %H:%M:%S", tm)
    return [scene, len(frames), objs, desc, 'night' if 'night' in desc else 'day', frames[0], tmstr, ego_pose['lat'], ego_pose['lng']]


for s in susc.get_scene_names():
    if not re.fullmatch(args.scenes, s):
        continue
    print(stat_one_scene(s))