import simplejson as json
import cv2
import numpy as np
import math
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as sp
import pythreejs as threejs
from pythreejs import *
import time
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from objectron.dataset import iou
from objectron.dataset import box

from scipy.spatial.transform import Rotation as R

objects_interest ='AlphabetSoup'

iou_sum = 0
iou_total = 0
iou_max = 0
iou_min = 1
def draw_boxes(boxes = [], clips = [], colors = ['r', 'b', 'g' , 'k']):
  """Draw a list of boxes.

      The boxes are defined as a list of vertices
  """
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111, projection='3d')
  for i, b in enumerate(boxes):
    x, y, z = b[:, 0], b[:, 1], b[:, 2]
    ax.scatter(x, y, z, c = 'r')
    for e in box.EDGES:
      ax.plot(x[e], y[e], z[e], linewidth=2, c=colors[i % len(colors)])

  if (len(clips)):
    points = np.array(clips)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=100, c='k')
    
  plt.gca().patch.set_facecolor('white')
  ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
  ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
  ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))

  # rotate the axes and update
  ax.view_init(30, 12)
  plt.draw()
  plt.show()

v1 = np.array([[-0.33968984, -0.05288385, -1.17507233],
 [-0.40124338,  0.05014238, -1.2934148],
 [-0.43014634, -0.06355323, -1.33895528],
 [-0.47653974,  0.04960908, -1.17049374],
 [-0.5054427,  -0.06408653, -1.21603422],
 [-0.17393698, -0.04168116, -1.13411044],
 [-0.20283994, -0.15537677, -1.17965092],
 [-0.24923333, -0.04221446, -1.01118938],
 [-0.27813629, -0.15591008, -1.05672986]])

v2 = np.array([[-0.28304723, -0.03790521, -0.95919561],
 [-0.34334973,  0.04826891, -1.07497346],
 [-0.35822576, -0.0410708,  -1.09632719],
 [-0.41593161,  0.03950843, -0.98775709],
 [-0.43080765, -0.04983126, -1.00911069],
 [-0.13528676, -0.02597914, -0.90928042],
 [-0.15016282, -0.11531886, -0.93063414],
 [-0.20786867, -0.03473962, -0.82206404],
 [-0.22274472, -0.12407933, -0.84341764]])

scale= -2.091109244641927
center = np.array([-0.18802881,  0.13143663, -0.937361 ])
normal = [-0.63777405, -0.07697788,  0.7663672 ]

# w1 = box.Box(vertices=v1)
# w2 = box.Box(vertices=v2)
# # Change the scale/position for testing
# b1 = box.Box.from_transformation(np.array(w1.rotation), np.array(w1.translation), np.array([3., 1., 0.5]))
# b2 = box.Box.from_transformation(np.array(w2.rotation), np.array(w2.translation), np.array([1.9, 1.6, 0.7]))
# # 0.3,
# loss = iou.IoU(b1, b2)
# print('iou = ', loss.iou())
# print('iou (via sampling)= ', loss.iou_sampling())
# intersection_points = loss.intersection_points
# draw_boxes([b1.vertices, b2.vertices],  clips=loss.intersection_points)

if __name__ == "__main__":

    import argparse
    import yaml 
    import glob 
    import os 

    parser = argparse.ArgumentParser()
    parser.add_argument("--outf",
        default="out_experiment",
        help='where to store the output')
    parser.add_argument("--data",
        default=None,
        help='folder for data images to load, *.png, *.jpeg, *jpg')
    opt = parser.parse_args()
    
    # load the images if there are some
    imgs = []
    imgs_out = []
    imgsname = []
    visibility_object = 0
    detect_object = 0
    if not opt.data is None:
        videopath = opt.data
        outpath = opt.outf 
        for j in sorted(glob.glob(videopath+"/*.png")):
            imgs.append(j)
            imgs_out.append(j.replace(videopath,outpath).replace(".png",".png.png"))
            imgsname.append(j.replace(videopath,"").replace("/",""))

    # starting the loop here
    i_image = -1 

    while True:
        i_image+=1  
        if i_image >= len(imgs):
            i_image =0
            break
            
        frame = cv2.imread(imgs[i_image])
        out = cv2.imread(imgs_out[i_image])
        img_name = imgsname[i_image]
 
        #frame = frame[...,::-1].copy()

        # load the json file
        all_projected_cuboid_keypoints = []
        all_location = []
        all_quaternion_xyzw = []
        flag = False
        inference_flag = False
        path_json = imgs[i_image].replace(".png",".json")


        path_json_out = imgs[i_image].replace(videopath,outpath).replace(".png",".json")
        with open(path_json) as f:
            data_json = json.load(f)


        # load the projected cuboid keypoints
        for obj in data_json['objects']:    
            if not objects_interest is None and \
                not obj['class'] in objects_interest\
                :
                continue
            #print(obj)
            # load the projected_cuboid_keypoints
            if obj['visibility'] == 1:
                flag = True
                visibility_object += 1
                projected_cuboid_keypoints = obj['projected_cuboid']
                location = obj["location"]
                quaternion_xyzw = obj["quaternion_xyzw"]
                quaternion_xyzw_world = obj["quaternion_xyzw_world"]
                # print(quaternion_xyzw)
                # box1.append(np.array([

                # ]))
                r_in = R.from_quat(quaternion_xyzw)
                #print(r.as_euler('zyx', degrees=True))

                o = r_in.as_euler('zyx', degrees=True)   
                r_fix = R.from_quat(quaternion_xyzw_world)
                world_o = r_fix.as_euler('zyx', degrees=True)   
                fix = o + world_o
                r2 = R.from_euler('zyx', fix, degrees=True)
                #print(r2.as_euler('zyx', degrees=True))

                # p0 = r_in.apply([-0.11035,0.06325,0.04215])
                # p1 = r_in.apply([-0.11035,0.06325,-0.04215])
                # p2 = r_in.apply([0.11035,0.06325,-0.04215])
                # p3 = r_in.apply([0.11035,0.06325,0.04215])
                # p4 = r_in.apply([-0.11035,-0.06325,0.04215])
                # p5 = r_in.apply([-0.11035,-0.06325,-0.04215])
                # p6 = r_in.apply([0.11035,-0.06325,-0.04215])
                # p7 = r_in.apply([0.11035,-0.06325,0.04215])

                
                p0 = r_in.apply([-0.11035,-0.06325,-0.04215])
                p1 = r_in.apply([0.11035,-0.06325,-0.04215])
                p2 = r_in.apply([0.11035,0.06325,-0.04215])
                p3 = r_in.apply([-0.11035,0.06325,-0.04215])
                p4 = r_in.apply([-0.11035,-0.06325,0.04215])
                p5 = r_in.apply([0.11035,-0.06325,0.04215])
                p6 = r_in.apply([0.11035,0.06325,0.04215])
                p7 = r_in.apply([-0.11035,0.06325,0.04215])

                p8 = [0,0,0]
                # print(np.array([p8,p5,p4,p0,p1,p6,p7,p2,p3]))
                box1=np.array([p8,p5,p4,p1,p0,p6,p7,p2,p3])
                # print(box1)
                # print(r.as_matrix())
                # print(r.apply([1,0,0]))
                
                # print(projected_cuboid_keypoints[0])
                # transfrom dope -> objrcton format



                # n = 0
                # for index in projected_cuboid_keypoints:
                #     x = int(index[0]) 
                #     y = int(index[1])
                #     #cv2.circle(frame,(int(index[0],int(index[1]) ),5,(0,0,255),-1 ))
                #     cv2.putText(frame, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                #     n+=1
                
            else:
                pass
                projected_cuboid_keypoints = [[-100,-100],[-100,-100],[-100,-100],\
                    [-100,-100],[-100,-100],[-100,-100],[-100,-100],[-100,-100],[-100,-100]]
            all_projected_cuboid_keypoints.append(projected_cuboid_keypoints)
        
        if len(all_projected_cuboid_keypoints) == 0:
            all_projected_cuboid_keypoints = [[[-100,-100],[-100,-100],[-100,-100],\
                    [-100,-100],[-100,-100],[-100,-100],[-100,-100],[-100,-100],[-100,-100]]]

        # flatten the keypoints
        flatten_projected_cuboid = []
        for obj in all_projected_cuboid_keypoints: 
            for p in obj:
                flatten_projected_cuboid.append(p)

        # load output dir
        with open(path_json_out) as f:
            data_json_out = json.load(f)
        for obj in data_json_out['objects']:  
            inference_flag = True
            quaternion_xyzw_out = obj["quaternion_xyzw"]
            # print(quaternion_xyzw_out)
            r_out = R.from_quat(quaternion_xyzw_out)
            detect_object +=1

            # rotation
            o = r_out.as_euler('zyx', degrees=True)  
            # print(o) 
            o = o +[5, -5, -5] 
            r_fix = R.from_euler('zyx', o, degrees=True)

            # use orginal or rotation
            r_fix = r_out

            # p0 = r_fix.apply([-0.11035,0.06325,0.04215])
            # p1 = r_fix.apply([-0.11035,0.06325,-0.04215])
            # p2 = r_fix.apply([0.11035,0.06325,-0.04215])
            # p3 = r_fix.apply([0.11035,0.06325,0.04215])
            # p4 = r_fix.apply([-0.11035,-0.06325,0.04215])
            # p5 = r_fix.apply([-0.11035,-0.06325,-0.04215])
            # p6 = r_fix.apply([0.11035,-0.06325,-0.04215])
            # p7 = r_fix.apply([0.11035,-0.06325,0.04215])
            p8 = [0,0,0] 

            p0 = r_in.apply([-0.11035,-0.06325,-0.04215]) + p8
            p1 = r_in.apply([0.11035,-0.06325,-0.04215])+ p8
            p2 = r_in.apply([0.11035,0.06325,-0.04215])+ p8
            p3 = r_in.apply([-0.11035,0.06325,-0.04215])+ p8
            p4 = r_in.apply([-0.11035,-0.06325,0.04215])+ p8
            p5 = r_in.apply([0.11035,-0.06325,0.04215])+ p8
            p6 = r_in.apply([0.11035,0.06325,0.04215])+ p8
            p7 = r_in.apply([-0.11035,0.06325,0.04215])+ p8



            
            #print(np.array([p8,p5,p4,p0,p1,p6,p7,p2,p3]))
            # box2= np.array([p8,p5,p4,p0,p1,p6,p7,p2,p3])
            box2= np.array([p8,p5,p4,p1,p0,p6,p7,p2,p3])
            # print(box2)
        if not inference_flag:
            continue
        w1 = box.Box(vertices=box1)
        w2 = box.Box(vertices=box2)
        loss = iou.IoU(w1, w2)
        if (flag):
            print('iou = ', loss.iou())
            if(iou_max<loss.iou()):
                iou_max = loss.iou()
            if(iou_min>loss.iou()):
                iou_min = loss.iou()
            # print(r_in.as_euler('zyx', degrees=True)-r_out.as_euler('zyx', degrees=True))
            iou_sum += loss.iou()
            iou_total +=1
        # print('iou (via sampling)= ', loss.iou_sampling())
        # intersection_points = loss.intersection_points

            #draw_boxes([w1.vertices, w2.vertices],  clips=loss.intersection_points)

        # b1 = box.Box.from_transformation(np.array(w1.rotation), np.array(w1.translation), np.array([3., 1., 0.5]))
        # b2 = box.Box.from_transformation(np.array(w2.rotation), np.array(w2.translation), np.array([1.9, 1.6, 0.7]))
        # loss = iou.IoU(b1, b2)
        # print('iou = ', loss.iou())
        # draw_boxes([b1.vertices, b2.vertices],  clips=loss.intersection_points)
        cv2.imshow("frame",frame)
        cv2.imshow("out",out)
        cv2.moveWindow("out", 600,55)
        key = cv2.waitKey(1)
        if key == ord('c'):
            #time.sleep(3)
            break
    iou_result = iou_sum/iou_total

    Accuracy = detect_object/visibility_object
    # print(iou_total) #194
    result = f'| visibility_object  | detect_object | Accuracy |\
             \n|:-------------------|:--------------|:---------|\
             \n|        {visibility_object}       |        {detect_object}       | {Accuracy:.5f} | '
    print(result)

    result = f'| 3D IOU Max. | 3D IOU Min. | 3D IOU Avg. |\
             \n|:------------|:------------|:------------|\
             \n|  {iou_max:.5f}    |  {iou_min:.5f}    |  {iou_result:.5f}    |'
    print(result)
    time.sleep(3)
    cv2.destroyAllWindows()