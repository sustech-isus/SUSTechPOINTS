








from suscape.dataset import SuscapeDataset

susc = SuscapeDataset('/home/lie/nas/suscape_scenes')
scenes = susc.get_scene_names()
for s in scenes:
    desc = susc.read_desc(s)
    obj_set = set()
    scene = susc.get_scene_info(s)

    for f in scene['frames']:
        objs = susc.read_label(s, f)
    
        for o in objs['objs']:
            obj_set.add(o['obj_type'])
    objs = list(obj_set)
    objs.sort()
    objstr = ','.join(objs)
    print(s, '|', desc['scene'], '|', objstr)