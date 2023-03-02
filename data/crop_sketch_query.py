import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
import os


def find_obj(arr):
   xmin, ymin, xmax, ymax = float('inf'), float('inf'), -1, -1
   list_pos =  np.argwhere(arr < 255)
   for pos in list_pos:
        i, j = pos[0], pos[1]
        ymax = max(i, ymax)
        ymin = min(i, ymin)
        xmax = max(j, xmax)
        xmin = min(j, xmin)

   return xmin, ymin, xmax, ymax

def sketch_crop(img, delta=30):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  h, w = gray.shape
  xmin, ymin, xmax, ymax = find_obj(gray)
  xmin = max(xmin - delta, 0)
  ymin = max(ymin - delta, 0)
  xmax = min(xmax + delta, w)
  ymax = min(ymax + delta, h)
  obj_img = img[ymin:ymax, xmin:xmax, :].copy()
  return obj_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
         description='Crop sketch image that focused on main object.')
    parser.add_argument('input', type=str, help='input directory path')
    parser.add_argument('output', type=str, help='output directory path')

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    for img in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img)
        new_img_path = os.path.join(output_dir, img)
        img = cv2.imread(img_path)
        img = sketch_crop(img)
        cv2.imwrite(new_img_path, img)



