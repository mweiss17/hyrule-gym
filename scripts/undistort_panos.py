import os
import cv2
import argparse
import yaml
import numpy as np


parser = argparse.ArgumentParser("Crops and rotates Vuze images")
parser.add_argument('config_file', type=str, help='Path to config file')
parser.add_argument('input_path', type=str, help='Path to folder containing the subfolders of images')
args = parser.parse_args()
input_path = args.input_path
config_file = args.config_file

def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat
yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)

# loading
with open(config_file) as fin:
    c = fin.read()
    # some operator on raw conent of c may be needed
    c = "%YAML 1.1"+os.linesep+"---" + c[len("%YAML:1.0"):] if c.startswith("%YAML:1.0") else c
    result = yaml.load(c, Loader=yaml.Loader)

cam = {}
cam['0'] = result['CamModel_V2_Set']['CAM_0']
cam['1'] = result['CamModel_V2_Set']['CAM_1']
cam['2'] = result['CamModel_V2_Set']['CAM_2']
cam['3'] = result['CamModel_V2_Set']['CAM_3']
cam['4'] = result['CamModel_V2_Set']['CAM_4']
cam['5'] = result['CamModel_V2_Set']['CAM_5']
cam['6'] = result['CamModel_V2_Set']['CAM_6']
cam['7'] = result['CamModel_V2_Set']['CAM_7']

img = cv2.imread(input_path)
shape = img.shape[0:2]
h = shape[0]
w = shape[1]
img = img[0:h, 0:int(w/3)]


for k,v in cam.items():
    K = v['K']
    D = np.array(v['DistortionCoeffs'])
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    v['D'] = D
    v['new_K'] = new_K

#img = cv2.resize(img, (w, h))[:, :, ::-1]
#img = img[crop_margin:h - crop_margin]

balance = 1
DIM = (w, h)
_img_shape = (h, w)

K = cam['0']['K']
D = cam['0']['D']
new_K = cam['0']['new_K']

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

cv2.imshow('original', cv2.resize(img,(int(img.shape[1]),int(img.shape[0]))))
cv2.imshow("rectified", undistorted_img)
cv2.waitKey(0)


'''os.chdir(input_path)
for folder in os.listdir():
    if not os.path.isdir(folder): continue
    if not (folder == 'image_1' or folder == 'image_2'): continue

    cam_num = folder.split('_')[-1]
    calib = cam[str(int(cam_num) -1)]
    if not os.path.isdir("../undistorted"): os.mkdir("../undistorted")
    if not os.path.exists("../undistorted/" + folder): os.mkdir("../undistorted/" + folder)

    for fname in os.listdir(folder):
        if os.path.isdir(folder + "/" + fname): continue
        out_fname = "../undistorted/" + folder + "/" + fname
        print("Processing: " + folder + "/" + fname)
        if os.path.isfile(out_fname): continue

        img = cv2.imread(folder+ '/' + fname)
        K = calib['K']
        D = calib['D']
        new_K = calib['new_K']

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        cv2.imwrite(out_fname, undistorted_img)

        cv2.imshow("original", cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2))))
        cv2.imshow("undistort", cv2.resize(img2,(int(img2.shape[1]/2),int(img2.shape[0]/2))))
        cv2.imshow("initUndistortRectifyMap", cv2.resize(undistorted_img,(int(undistorted_img.shape[1]/2),int(undistorted_img.shape[0]/2))))
        cv2.waitKey(0)'''