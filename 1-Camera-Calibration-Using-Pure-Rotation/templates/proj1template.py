#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Proj 1, Calibration using a sequence of purely rotated images
from scipy.io import loadmat
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

xxx = loadmat('../data/pureRotPrexyCorrespondencePoints.mat')
x1pMat = xxx['x1pMat']
x2pMat = xxx['x2pMat']

fn = '../data/prexy'
for i in range(1, 6):
    I1 = cv2.imread(fn + str(i) + '.jpg')
    I2 = cv2.imread(fn + str(i + 1) + '.jpg')
    cv2.imshow("Image 1", I1)
    cv2.imshow("Image 2", I2)
    xtemp = x1pMat[:, :, i - 1]
    n = 0
    while (linalg.norm(xtemp[:, n]) > 0 and n < 99):
        n = n + 1
    n = n - 1
    # extract pixel correspondence points in first image
    x1pmat = x1pMat[:, 0:n, i - 1]
    # extract pixel correspondence points in second image
    x2pmat = x2pMat[:, 0:n, i - 1]
    cv2.waitKey(0)

# create some fake reprojections so you can see how to do plots
# you'll need to implement the calculations for the actual reprojections
x1pReproMat = x1pmat  #perfect reprojections (fake)
x2pReproMat = x2pmat

nn = x1pmat.shape
n = nn[1]

for i in range(0, n):
    cv2.putText(I1, "{}".format("X"), (int(x1pmat[0, i]), int(x1pmat[1, i])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
    cv2.putText(I2, "{}".format("X"), (int(x2pmat[0, i]), int(x2pmat[1, i])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
    # add reprojections to image
    cv2.putText(I1, "{}".format("O"),
                (int(x1pReproMat[0, i]), int(x1pReproMat[1, i])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 2)
    cv2.putText(I2, "{}".format("O"),
                (int(x2pReproMat[0, i]), int(x2pReproMat[1, i])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 2)

cv2.imshow("Image 1 plus correspondences", I1)
cv2.imshow("Image 2 plus correspondences", I2)
cv2.waitKey(0)
