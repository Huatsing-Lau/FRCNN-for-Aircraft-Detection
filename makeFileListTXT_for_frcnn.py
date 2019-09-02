# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 20:51:03 2018
本文件的作用是为DOTA2018的官方数据生成适合frcnn_kitti训练所需的txt格式的人工标注文件
@author: liuhuaqing
"""



#import read_bbox

import matplotlib.pyplot as plt
import cv2 as cv
import os
import numpy as np



def my_round(bbox):
  if len(bbox)==2:
    return tuple(map(round, bbox))
  else:
    return list(map(my_round, bbox))

# 在原图上画bbox
def draw_bboxes_and_labels_on_img(imgfilename,txtfilename,dstdir):
  #get labels and bboxes
  labels, bboxes = get_labels_from_txt(txtfilename)
  #-1 represent unchanged depth read model
  img = cv.imread(imgfilename, -1)

  #bboxes = map(my_round, bboxes)
  for bbox in bboxes:
    cv.line(img, (bbox[0],bbox[1]), (bbox[0],bbox[3]), (0,255,0),3)
    cv.line(img, (bbox[0],bbox[3]), (bbox[2],bbox[3]), (0,255,0),3)
    cv.line(img, (bbox[2],bbox[3]), (bbox[2],bbox[1]), (0,255,0),3)
    cv.line(img, (bbox[2],bbox[1]), (bbox[0],bbox[1]), (0,255,0),3)

  name = imgfilename.split('\\')[-1][:-4]
  cv.imwrite( os.path.join(dstdir, name+"_labelled.tif"), img )


#return labels and bboxes read from txt file
def get_labels_from_txt(txtfile_name):
  
  labels = []
  bboxes = []

  with open(txtfile_name, 'r') as f:
      print('Parsing annotation files: '+txtfile_name)
      for line in f:
          line_split = line.strip().split(' ')#strip()用于去除尾部的回车符
          (x1, y1, x2, y2, class_name,difficult) = line_split
          bboxes.append([int(float(x1)),int(float(y1)),int(float(x2)),int(float(y2))])
          labels.append(class_name)

  return labels, bboxes  


# 生成DOTA2018数据适用于keras-frcnn训练所需的txt文件
def makeTxtFrcnn(labels, bboxes, image_filename, output_txt_filename):
    list_file = open(output_txt_filename, 'a+')#读写模式的设定参考：https://blog.csdn.net/yaoxy/article/details/79441462  
    for (bbox,label) in zip(bboxes,labels):
        x1 = int(float(bbox[0]))
        y1 = int(float(bbox[1]))
        x2 = int(float(bbox[2]))
        y2 = int(float(bbox[3]))

        list_file.write( '{},{},{},{},{},{}\n'.format(image_filename,x1,y1,x2,y2,label) )
    
    list_file.close()
    return


def mkFileList_for_frcnn_Xingtubei(txt_srcDir,img_srcDir,output_txt_filename):
    for txt,img in zip(os.listdir(txt_srcDir),os.listdir(img_srcDir)):
        if txt.split(".")[-1] == "txt" and img.split(".")[-1] in ["png","jpg","tif"]:
            txt_filename = os.path.join(txt_srcDir,txt)
            #img_filename = os.path.join(img_srcDir,img)
            img_filename = os.path.join(img_srcDir,txt[:-3]+img[-3:])
            print('processing:  '+ txt)
            labels, bboxes = get_labels_from_txt( txt_filename )
            makeTxtFrcnn(labels, bboxes, img_filename, output_txt_filename)
    
def draw_labels_on_images_frcnn_Xingtubei(txt_srcDir,img_srcDir,dstdir):
    for txt,img in zip(os.listdir(txt_srcDir),os.listdir(img_srcDir)):
        if txt.split(".")[-1] == "txt" and img.split(".")[-1] in ["png","jpg","tif"]:
            print('processing: '+txt)
            img = txt[:-3]+img.split(".")[-1]
            draw_bboxes_and_labels_on_img(os.path.join(img_srcDir,img),os.path.join(txt_srcDir,txt),dstdir)

    
if __name__ == "__main__":
    
    txt_srcDir = '../DOTA_devkit/xingtubeisplit_aircraft_only_txt_Rect'
    img_srcDir = '../DOTA_devkit/xingtubeisplit/images'#
    dstdir = os.getcwd()
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
    output_txt_filename = os.path.join(dstdir, "DOTA2018_OpticalAircraft_bboxes.txt")#"Chreoc_OpticalAircraft_bboxes.txt"
    
    if os.path.exists(output_txt_filename):
        os.remove(output_txt_filename)
    
    mkFileList_for_frcnn_Xingtubei(txt_srcDir,img_srcDir,output_txt_filename)
    
# =============================================================================
#     dstdir = '..\\DOTA_devkit\\xingtubeisplit\\images_labelled'
#     draw_labels_on_images_frcnn_Xingtubei(txt_srcDir,img_srcDir,dstdir)
# =============================================================================
    
# =============================================================================
#     imgfilename = 'E:\\Xingtubei\\keras_Faster-RCNN_xingtubei\\DOTA_devkit\\xingtubeisplit\\images\\P0000__0.25__0___512.png'
#     txtfilename = 'E:\\Xingtubei\\keras_Faster-RCNN_xingtubei\\DOTA_devkit\\xingtubeisplit_aircraft_only_txt_Rect\\P0000__0.25__0___512.txt'
#     dstdir = 'E:\\Xingtubei\\keras_Faster-RCNN_xingtubei\\DOTA_devkit\\xingtubeisplit\\images_labelled'
#     draw_bboxes_and_labels_on_img(imgfilename,txtfilename,dstdir)
# =============================================================================
            
            
