import read_bbox
import matplotlib.pyplot as plt
import cv2 as cv
import os
import numpy as np

cwd = os.getcwd()
DIR = os.path.join(cwd, "plane\data")#图片和xml地址

def my_round(bbox):
  if len(bbox)==2:
    return tuple(map(round, bbox))
  else:
    return list(map(my_round, bbox))

def label(filename):
  #get labels and bboxes
  labels, bboxes = read_bbox.bbox(filename.split(".")[0]+".xml")
  #-1 represent unchanged depth read model
  img = cv.imread(filename, -1)

  bboxes = map(my_round, bboxes)
  for bbox in bboxes:
    cv.line(img, bbox[0], bbox[1], (0,255,0),3)
    cv.line(img, bbox[1], bbox[2], (0,255,0),3)
    cv.line(img, bbox[2], bbox[3], (0,255,0),3)
    cv.line(img, bbox[3], bbox[0], (0,255,0),3)

  #save image
  cv.imwrite(filename.split(".")[0]+"_labelled.tif", img)

if __name__ == "__main__":
  for file in os.listdir(DIR):
    if file.split(".")[1] == "tif":
      label(os.path.join(DIR, file))
