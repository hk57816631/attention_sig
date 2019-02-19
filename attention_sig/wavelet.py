from __future__ import print_function

import os

import scipy.io

from PIL import Image

import numpy as np

import argparse

import cv2

import pywt

img_path = "./image/BUS/"#args.img_path

result_path = './image/wavelet/'

test_file = './image/tumor.txt'

with open(test_file,"r") as f:
    ls = f.readlines()
namesimg = [l.rstrip('\n') for l in ls]
nb_data_img = len(namesimg)

for i in range(nb_data_img):

    Xpath = img_path + "{}.bmp".format(namesimg[i])

    print(Xpath)

    # # wavelet image generation

    img = cv2.imread(Xpath, 0)

    equ = cv2.equalizeHist(img,256)

    coeffs = pywt.dwt2(equ, 'haar')

    cA, (cH, cV, cD) = coeffs

    C0 = cv2.resize(cA,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LINEAR)

    C1 = cv2.resize(cH,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LINEAR)

    C2 = cv2.resize(cV,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LINEAR)

    C3 = cv2.resize(cD,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_LINEAR)

    wavelet2 = np.square(C1) + np.square(C2)

    wavelet= np.sqrt(wavelet2)

    wavelet_new = np.zeros((img.shape[0],img.shape[1],3))



    C0min, C0max = C0.min(), C0.max()

    C01 = ((C0-C0min)/(C0max-C0min))*255

    waveletmin, waveletmax = wavelet.min(), wavelet.max()

    wavelet1 = ((wavelet-waveletmin)/(waveletmax-waveletmin))*255

    wavelet_new[:,:,0] = equ

    wavelet_new[:,:,1] = C01

    wavelet_new[:,:,2] = wavelet1

    wavelet_new = np.uint8(wavelet_new)

    imgorg = Image.fromarray(wavelet_new)

    b = "{}.bmp".format(namesimg[i])

    imgorg.save(result_path + b)
