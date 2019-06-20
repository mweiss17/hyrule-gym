"""Crops and rotates Vuze images"""

import os
import cv2
import argparse
import multiprocessing as mp
from multiprocessing import Queue

from datetime import datetime
startTime = datetime.now()

def face_proc(q):
    while True:
        try:
            fname = q.get(True, 1)
            print('Processing : ' + input_path + '/' + fname)
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            H, W = img.shape[:2]

            faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(1,1),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )
            for (x, y, w, h) in faces:
                    res = cv2.blur(img[y:y+h, x:x+w] , (35,35))
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    img[y:y+h, x:x+w] = res

            out_fname = "../" + output_folder + '/' + fname
            if os.path.isfile(out_fname):
                    continue
            cv2.imwrite(out_fname, img)
                                
        except:
            return

parser = argparse.ArgumentParser("Detects faces in panos")
parser.add_argument('input_path', type=str, help='Path to folder containing the images')
parser.add_argument('output_folder', type=str, help='Output folder')
args = parser.parse_args()

print("Path : " + args.input_path)
if not os.path.exists(args.input_path):
    parser.error("Input folder does not exist or does not contain any images")
print(args.input_path)

input_path = args.input_path
output_folder = args.output_folder
os.chdir(input_path)
if not os.path.exists("../" + output_folder):
    os.mkdir("../" + output_folder)

cascPath = "../haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

total_frames = len(os.listdir())
q = Queue()

num_procs = 8
procs = []

for i in range(num_procs):
    p = mp.Process(target=face_proc, args=(q,))
    procs.append(p)
    p.start()

fnames = [fname for fname in os.listdir() if fname != ".DS_Store"]
while fnames:
    if q.empty():
        q.put(fnames.pop())

for p in procs:
    p.join()

print(datetime.now() - startTime)
