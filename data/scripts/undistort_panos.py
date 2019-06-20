"""Crops top and bottom margins from images"""

import os
import cv2
import yaml
import numpy as np
import argparse

#parser = argparse.ArgumentParser("Crops top and bottom margins")
#parser.add_argument('input_path', type=str, help='Path to folder containing the images')
#parser.add_argument('new_H', type=int, help='Desired image height')
#args = parser.parse_args()

#input_path = args.input_path
#new_H = args.new_H
#os.chdir(input_path)

def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat

yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)

with open('vuze_config/VZP1186200216.yml') as fin:
    c = fin.read()
    # some operator on raw conent of c may be needed
    c = "%YAML 1.1"+os.linesep+"---" + c[len("%YAML:1.0"):] if c.startswith("%YAML:1.0") else c
    result = yaml.load(c, Loader=yaml.Loader)

cam = result['CamModel_V2_Set']['CAM_0']

W = int(3840)
H = int(1920)
new_H = int(1320)
crop_W = int(new_H*1088/1600)

h = new_H
w = crop_W
balance = 1
DIM = (w, h)
_img_shape = (h, w)

K = cam['K']
D = np.array(cam['DistortionCoeffs'])
new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))

fname = "2019-06-10/panoramas/01/pano_000001.jpg"
img = cv2.imread(fname)
img = img[int(200):int(H - 400), 0:W]

crops = []
for i in range(16):
	crop = img[0:new_H, int(i*(W/16)):int(i*(W/16) + crop_W)]
	map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)
	undistorted_img = cv2.remap(crop, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
	crops.append(undistorted_img)

cv2.imshow("pano", cv2.resize(img, (int(W/2), int(new_H/2))))

for i, undistort in enumerate(crops):
	cv2.imshow("crop_"+ str(i), cv2.resize(undistort, (int(crop_W/2), int(new_H/2))))
	
cv2.waitKey(0)

'''for idx, fname in enumerate(os.listdir1()):
    if fname == ".DS_Store": continue
    if not fname.split('.')[-1] == 'png': continue

    print('Processing: ' + fname + " --- " + str(idx))
    img = cv2.imread(fname)
    crop_img = img[int(H/2 - new_H/2):int(H/2 + new_H/2), 0:W]
    out_fname = '../crop/' + fname
    cv2.imwrite(out_fname, crop_img)'''

