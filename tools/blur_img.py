import cv2
import numpy as np
import json 
import os
# Load an image

def proc_one_img(imgfile, labelfile, save_path):
    im_arr = cv2.imread(imgfile) #"/home/lie/nas/suscape_scenes/scene-000219/camera/front_right/1632795409.500.jpg")


    if os.path.exists(labelfile):
        with open(labelfile) as f: #"/home/lie/nas/scenes_camera/scene-000219/label/camera/front_right/1632795409.500.json") as f:
            labels = json.load(f)

        
        objs = labels['objs']
        for obj in objs:
            x1 = int(obj['rect']['x1'])
            y1 = int(obj['rect']['y1'])
            x2 = int(obj['rect']['x2'])
            y2 = int(obj['rect']['y2'])

            if x2-x1 <= 0 or y2-y1 <= 0:
                pass
            else:

                ksize = int(np.max([5, (x2-x1)/5, (y2-y1)/5]))
                ksize = ksize + 1 if ksize % 2 == 0 else ksize
                im_arr[y1:y2, x1:x2] = cv2.medianBlur(im_arr[y1:y2, x1:x2], 
                                                    ksize)
    else:
        print("label file not found", labelfile)

    #cv2.imshow('after',im_arr)
    cv2.imwrite(save_path, im_arr)



rootdir = '/home/lie/nas/scenes_camera'
save_dir = '/home/lie/nas/suscape_blured'
# scenes = os.listdir(rootdir)
scenes = ['scene-000812']
scenes.sort()

#237, 669,
for scene in scenes:
    scene_path = os.path.join(rootdir, scene, 'camera')
    if not os.path.isdir(scene_path):
        continue
    print(scene_path)
    for camera in os.listdir(scene_path):
        camera_path = os.path.join(scene_path, camera)
        if not os.path.isdir(camera_path):
            continue
        print(camera_path)
        for frame in os.listdir(camera_path):
            image_path = os.path.join(camera_path, frame)
            if not os.path.splitext(image_path)[1] == '.jpg':
                continue

            label_path = os.path.join(rootdir, scene, 'label', 'camera', camera,  os.path.splitext(frame)[0]+'.json')

            save_path = os.path.join(save_dir, scene, camera, frame)
            os.makedirs(os.path.join(save_dir, scene, camera), exist_ok=True)

            proc_one_img(image_path, label_path, save_path)
            #print(image_path)
            # img = os.path.join(frame_path, os.listdir(frame_path)[0])
            # label = os.path.join(frame_path, os.listdir(frame_path)[1])
            # save_path = os.path.join(frame_path, os.listdir(frame_path)[0])
            # proc_one_img(img, label, save_path)
            #print(img)
            #print(label)
            #print(save_path)
            #break
        #break
    #break
