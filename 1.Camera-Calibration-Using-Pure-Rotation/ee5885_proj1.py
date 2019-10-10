#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ee5885_proj1.py - 2019-10-10 15:07
#
# Copyright © 2019 Libao Jin <jinlibao@outlook.com>
# Distributed under terms of the MIT license.
#
'''
Ee5885_Proj1
'''

import cv2
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.backends.backend_pdf import PdfPages


class Ee5885_Proj1(object):

    def __init__(self):
        plt.style.use('ggplot')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    def plot(self, x, y, filename):
        with PdfPages(filename) as pdf:
            fig = plt.figure()
            plt.plot(x, y)
            plt.show(block=False)
            pdf.savefig(fig)

    def read_data(self, filename):
        mat = loadmat(filename)
        return (mat['x1pMat'], mat['x2pMat'])

    def calibrate(self):
        pass

    def reproject(self, x, R=[], K=[]):
        return x

    def test(self):
        mat_file = './data/pureRotPrexyCorrespondencePoints.mat'
        x1pMat, x2pMat = self.read_data(mat_file)
        I = [cv2.imread('./data/prexy{:d}.jpg'.format(i + 1)) for i in range(7)]
        ncols = []
        for i in range(len(I) - 1):
            cv2.imshow("Image 1 (prexy{:d}.jpg)".format(i + 1), I[i])
            cv2.imshow("Image 2 (prexy{:d}.jpg)".format(i + 2), I[i + 1])
            xtemp = x1pMat[:, :, i]
            for j in range(xtemp.shape[1]):
                if sp.linalg.norm(xtemp[:, j]) == 0 or j == xtemp.shape[1] - 1:
                    ncols.append(j)
                    break
            cv2.waitKey(0)

        for i in range(len(I) - 1):
            x1pmat = x1pMat[:, 0:ncols[i], i]
            x2pmat = x2pMat[:, 0:ncols[i], i]
            x1pReproMat = self.reproject(x1pmat)
            x2pReproMat = self.reproject(x2pmat)
            for j in range(0, x1pmat.shape[1]):
                cv2.putText(I[i], "{}".format("X"),
                            (int(x1pmat[0, j]), int(x1pmat[1, j])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
                cv2.putText(I[i + 1], "{}".format("X"),
                            (int(x2pmat[0, j]), int(x2pmat[1, j])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
                # add reprojections to image
                cv2.putText(I[i], "{}".format("O"),
                            (int(x1pReproMat[0, j]), int(x1pReproMat[1, j])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 2)
                cv2.putText(I[i + 1], "{}".format("O"),
                            (int(x2pReproMat[0, j]), int(x2pReproMat[1, j])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 2)
            cv2.imshow("Image 1 reproj (prexy{:d}.jpg)".format(i + 1), I[i])
            cv2.imshow("Image 2 reproj (prexy{:d}.jpg)".format(i + 2), I[i + 1])
            cv2.waitKey(0)


if __name__ == '__main__':
    c = Ee5885_Proj1()
    c.test()
'''
End of file
'''
