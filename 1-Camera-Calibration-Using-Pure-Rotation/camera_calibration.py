#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# camera_calibration.py - 2019-10-10 15:07
#
# Copyright © 2019 Libao Jin <jinlibao@outlook.com>
# Distributed under terms of the MIT license.
#
'''
Camera Calibration
'''

import os
import numpy as np
import scipy as sp
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.backends.backend_pdf import PdfPages
import cv2


class CameraCalibration():
    '''Camera Calibratoin Using Pure Rotation'''

    def __init__(self):
        plt.style.use('ggplot')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        np.random.seed(seed=888)

    def read_data(self, filename):
        mat = loadmat(filename)
        return (mat['x1pMat'], mat['x2pMat'])

    def read_csv(self, output_dir):
        R_list = []
        for i in range(6):
            R = np.loadtxt('{:s}/R_{:d}.csv'.format(output_dir, i + 1),
                           delimiter=',')
            R_list.append(R)
        K = np.loadtxt('{:s}/K.csv'.format(output_dir), delimiter=',')
        return (R_list, K)

    def print(self, A, name='A', filename=''):
        nrow, ncol = A.shape
        if len(filename) == 0:
            print(r'\begin{equation*}')
            print('{:s} = \n'.format(name))
            print(r'\begin{bmatrix}')
            for i in range(nrow):
                print(r'{:8.4f} & {:8.4f} & {:8.4f} \\'.format(
                    A[i, 0], A[i, 1], A[i, 2]))
            print(r'\end{bmatrix}')
            print(r'\end{equation*}')
            return

        with open(filename, 'w') as f:
            f.write('\\begin{equation*}\n')
            f.write('{:s} = \n'.format(name))
            f.write('\\begin{bmatrix}\n')
            for i in range(nrow):
                f.write('{:8.4f} & {:8.4f} & {:8.4f} \\\\\n'.format(
                    A[i, 0], A[i, 1], A[i, 2]))
            f.write('\\end{bmatrix}.\n')
            f.write('\\end{equation*}\n')

    def print_matrix(self, output_dir):
        R_list, K = self.read_csv(output_dir)
        for i in range(len(R_list)):
            self.print(R_list[i], '\\widetilde{{R}}_{{{:d}}}'.format(i + 1),
                       '{:s}/R_{:d}.tex'.format(output_dir, i + 1))
        self.print(K, 'K', '{:s}/K.tex'.format(output_dir))

    def show_image(self, I, x, xr, filename):
        with PdfPages(filename) as pdf:
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            if I.ndim == 2:
                plt.imshow(I, cmap='gray')
            else:
                plt.imshow(I)
            h1, = plt.plot(x[0, :],
                           x[1, :],
                           '.',
                           markersize=4,
                           label='original points')
            h2, = plt.plot(xr[0, :],
                           xr[1, :],
                           'x',
                           markersize=4,
                           label='reprojection')
            plt.legend(handles=[h1, h2], loc='lower right')
            plt.grid()
            plt.axis([-0.5, 1279.5, 959.5, -0.5])
            plt.axis('off')
            plt.show(block=False)
            plt.savefig(filename.replace('pdf', 'png'), format='png', dpi=300)
            pdf.savefig(fig)
            plt.close()

    def hat(self, v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def D(self, z, v):
        D = np.zeros((4, 4))
        D[0, 1:] = self.hat(z).dot(v).T
        D[1:, 0] = self.hat(z).dot(v).reshape(3)
        D[1:, 1:] = z.dot(v.T) + v.dot(z.T) - 2 * z.T.dot(v) * np.eye(3)
        return D

    def eigvec(self, A):
        a, w = np.linalg.eig(A)
        return w[:, 0].reshape((4, 1))

    def q2R(self, q4):
        q0 = q4[0, 0]
        q = q4[1:, 0]
        R = np.eye(3) + 2 * q0 * self.hat(q) + 2 * self.hat(q).dot(self.hat(q))
        return R

    def optimal_quaternion(self, x1, x2):
        D = np.zeros((4, 4))
        for i in range(len(x1)):
            z, v = x1[i].reshape(3, 1), x2[i].reshape(3, 1)
            D += self.D(z, v)
        q = self.eigvec(D)
        R = self.q2R(q)
        return R

    def K(self, k):
        k = k.reshape(5, 1)
        return np.array([[k[0, 0], k[1, 0], k[2, 0]], [0, k[3, 0], k[4, 0]],
                         [0, 0, 1]])

    def invert_K(self, k):
        k = k.reshape(5, 1)
        return np.array([[
            1 / k[0, 0], -k[1, 0] / (k[0, 0] * k[3, 0]),
            k[1, 0] * k[4, 0] / (k[0, 0] * k[3, 0]) - k[2, 0] / k[0, 0]
        ], [0, 1 / k[3, 0], -k[4, 0] / k[3, 0]], [0, 0, 1]])

    def find_k(self, x1, x2, tol, max_iter):
        R = self.find_R(x1, x2, np.random.rand(5, 1))
        K = lambda k: self.K(k)
        K_inv = lambda k: self.invert_K(k)
        x1_reproj = lambda R, k: np.array([
            np.linalg.norm(x2[:, i] - K(k).dot(R.T.dot(K_inv(k).dot(x2[:, i]))))
            for i in range(x2.shape[1])
        ]) / x2.shape[1]
        x2_reproj = lambda R, k: np.array([
            np.linalg.norm(x1[:, i] - K(k).dot(R.dot(K_inv(k).dot(x1[:, i]))))
            for i in range(x1.shape[1])
        ]) / x1.shape[1]
        J = lambda k: np.sum(x1_reproj(R, k)) + np.sum(x2_reproj(R, k))
        k = np.random.rand(5, 1)
        i = 0
        while J(k) > tol and i < max_iter:
            print('{:3d}: {:.4f}'.format(i, J(k)))
            R = self.find_R(x1, x2, k)
            J = lambda k: np.sum(x1_reproj(R, k)) + np.sum(x2_reproj(R, k))
            y = minimize(J, k.T)
            k = y['x'].reshape(5, 1)
            i += 1
        print('{:3d}: {:.4f}'.format(i, J(k)))
        return k

    def find_R(self, x1, x2, k):
        K_guess = self.K(k)
        K_inv_guess = np.linalg.inv(K_guess)
        X1 = [K_inv_guess.dot(x1[:, i]) for i in range(x1.shape[1])]
        X2 = [K_inv_guess.dot(x2[:, i]) for i in range(x2.shape[1])]
        R = self.optimal_quaternion(X1, X2)
        return R

    def find_R_K(self, x1, x2, tol, max_iter):
        k = self.find_k(x1, x2, tol, max_iter)
        R = self.find_R(x1, x2, k)
        K = self.K(k)
        return (R, K)

    def find_all_R(self, x1_list, x2_list, tol, max_iter):
        R_list = []
        for i in range(len(x1_list)):
            print('Finding R between Image {:d} and Image {:d}...'.format(
                i, i + 1))
            R, K = self.find_R_K(x1_list[i], x2_list[i], tol, max_iter)
            R_list.append(R)
        return R_list

    def calibrate(self, x1_list, x2_list, tol, max_iter, output_dir):
        R_list = self.find_all_R(x1_list, x2_list, tol, max_iter)
        K = lambda k: self.K(k)
        K_inv = lambda k: self.invert_K(k)
        x1_reproj = lambda R, k, j: np.array([
            np.linalg.norm(x2_list[j][:, i] - K(k).dot(
                R.T.dot(K_inv(k).dot(x2_list[j][:, i]))))
            for i in range(x2_list[j].shape[1])
        ]) / x2_list[j].shape[1]
        x2_reproj = lambda R, k, j: np.array([
            np.linalg.norm(x1_list[j][:, i] - K(k).dot(
                R.T.dot(K_inv(k).dot(x1_list[j][:, i]))))
            for i in range(x1_list[j].shape[1])
        ]) / x1_list[j].shape[1]
        J = lambda k: np.sum([
            np.sum(x1_reproj(R_list[j], k, j)) + np.sum(
                x2_reproj(R_list[j], k, j)) for j in range(len(x1_list))
        ])
        k = np.random.rand(5, 1)
        print('Finding the calibration matrix K for all rotations...')
        y = minimize(J, k)
        k = y['x']
        for i in range(len(R_list)):
            R = R_list[i]
            np.savetxt('{:s}/R_{:d}.csv'.format(output_dir, i + 1),
                       R,
                       delimiter=',',
                       fmt='%15.8f',
                       newline='\n')
        np.savetxt('{:s}/K.csv'.format(output_dir, i + 1),
                   self.K(k),
                   delimiter=',',
                   fmt='%15.8f',
                   newline='\n')
        return (R_list, self.K(k))

    def reproject(self, x, R=np.eye(3), K=np.eye(3)):
        K_inv = np.linalg.inv(K)
        x_reproj = K.dot(R.dot(K_inv.dot(x)))
        return x_reproj

    def solve(self, mat_file, images, output_dir, tol=1, max_iter=200):
        I = [img.imread(image) for image in images]
        x1pMat, x2pMat = self.read_data(mat_file)
        ncols = []
        for i in range(len(I) - 1):
            for j in range(x1pMat[:, :, i].shape[1]):
                if sp.linalg.norm(x1pMat[:, j, i]
                                 ) == 0 or j == x1pMat[:, :, i].shape[1] - 1:
                    ncols.append(j)
                    break
        x1_list = [x1pMat[:, 0:ncols[i], i] for i in range(len(ncols))]
        x2_list = [x2pMat[:, 0:ncols[i], i] for i in range(len(ncols))]
        if (os.path.exists('{:s}/K.csv'.format(output_dir))):
            R_list, K = self.read_csv(output_dir)
        else:
            R_list, K = self.calibrate(x1_list, x2_list, tol, max_iter,
                                       output_dir)
        for i in range(len(R_list)):
            print('Rotation from Image {:d} to Image {:d}:'.format(
                i + 1, i + 2))
            print(R_list[i])
        print('Calibration matrix:')
        print(K)
        self.print_matrix(output_dir)

        for i in range(len(I) - 1):
            I1, I2 = I[i].copy(), I[i + 1].copy()
            x1pmat = x1pMat[:, 0:ncols[i], i]
            x2pmat = x2pMat[:, 0:ncols[i], i]
            x1pReproMat = self.reproject(x1pmat, R_list[i], K)
            x2pReproMat = self.reproject(x2pmat, R_list[i], K)
            filename_1 = '{:s}/prexy{:d}_2.pdf'.format(output_dir, i + 1)
            filename_2 = '{:s}/prexy{:d}_1.pdf'.format(output_dir, i + 2)
            self.show_image(I1, x1pmat, x1pReproMat, filename_1)
            self.show_image(I2, x2pmat, x2pReproMat, filename_2)

    def solve_cv2(self, mat_file, images, output_dir, tol=16, max_iter=100):
        I = [cv2.imread(image) for image in images]
        x1pMat, x2pMat = self.read_data(mat_file)
        ncols = []
        for i in range(len(I) - 1):
            for j in range(x1pMat[:, :, i].shape[1]):
                if sp.linalg.norm(x1pMat[:, j, i]
                                 ) == 0 or j == x1pMat[:, :, i].shape[1] - 1:
                    ncols.append(j)
                    break
        x1_list = [x1pMat[:, 0:ncols[i], i] for i in range(len(ncols))]
        x2_list = [x2pMat[:, 0:ncols[i], i] for i in range(len(ncols))]
        if (os.path.exists('{:s}/K.csv'.format(output_dir))):
            R_list, K = self.read_csv(output_dir)
        else:
            R_list, K = self.calibrate(x1_list, x2_list, tol, max_iter,
                                       output_dir)
        for i in range(len(R_list)):
            print('Rotation from Image {:d} to Image {:d}:'.format(
                i + 1, i + 2))
            print(R_list[i])
        print('Calibration matrix:')
        print(K)
        self.print_matrix(output_dir)

        for i in range(len(I) - 1):
            I1, I2 = I[i].copy(), I[i + 1].copy()
            x1pmat = x1pMat[:, 0:ncols[i], i]
            x2pmat = x2pMat[:, 0:ncols[i], i]
            x1pReproMat = self.reproject(x1pmat, R_list[i], K)
            x2pReproMat = self.reproject(x2pmat, R_list[i], K)
            for j in range(0, x1pmat.shape[1]):
                cv2.putText(I1, "{}".format("X"),
                            (int(x1pmat[0, j]), int(x1pmat[1, j])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
                cv2.putText(I2, "{}".format("X"),
                            (int(x2pmat[0, j]), int(x2pmat[1, j])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
                cv2.putText(I1, "{}".format("O"),
                            (int(x1pReproMat[0, j]), int(x1pReproMat[1, j])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 2)
                cv2.putText(I2, "{}".format("O"),
                            (int(x2pReproMat[0, j]), int(x2pReproMat[1, j])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 2)
            cv2.imshow("Image 1 reproj (prexy{:d}.jpg)".format(i + 1), I1)
            cv2.imshow("Image 2 reproj (prexy{:d}.jpg)".format(i + 2), I2)
            cv2.imwrite('{:s}/prexy{:d}_2.jpg'.format(output_dir, i + 1), I1)
            cv2.imwrite('{:s}/prexy{:d}_1.jpg'.format(output_dir, i + 2), I2)
            cv2.waitKey(300)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    c = CameraCalibration()
    mat_file = './data/pureRotPrexyCorrespondencePoints.mat'
    output_dir = './output'
    images = ['./data/prexy{:d}.jpg'.format(i + 1) for i in range(7)]
    c.solve(mat_file, images, output_dir, 1, 200)
    # c.solve_cv2(mat_file, images, output_dir, 1, 200)
'''
End of file
'''
