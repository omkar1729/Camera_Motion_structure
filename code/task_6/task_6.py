import cv2
import numpy as np
from cv2 import aruco

objp = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
print(objp.shape)

intrinsic_matrix_left = np.loadtxt('../../parameters/intrinsic_l.csv', delimiter=',')
intrinsic_matrix_right = np.loadtxt('../../parameters/intrinsic_r.csv', delimiter=',')
distortion_left = np.loadtxt('../../parameters/distortion_l.csv', delimiter=',')
distortion_right = np.loadtxt('../../parameters/distortion_r.csv', delimiter=',')

for i in range(11):
    left_img = cv2.imread('../../images/task_6/left_' + str(i) + '.png')
    right_img = cv2.imread('../../images/task_6/right_' + str(i) + '.png')

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    corners_l, ids_l, rejectedImgPoints_l = aruco.detectMarkers(left_img, aruco_dict)
    corners_r, ids_r, rejectedImgPoints_r = aruco.detectMarkers(right_img, aruco_dict)

    frame_markers_l = aruco.drawDetectedMarkers(left_img.copy(), corners_l)
    frame_markers_r = aruco.drawDetectedMarkers(right_img.copy(), corners_r)

    corners_l = np.asarray(corners_l)
    corners_r = np.asarray(corners_r)

    corners_l = np.squeeze(corners_l, axis=0)
    corners_r = np.squeeze(corners_r, axis=0)
    print(corners_l.shape)

    _, rvec_l, tvec_l = cv2.solvePnP(objp, corners_l, intrinsic_matrix_left, distortion_left)
    rvec_l, _ = cv2.Rodrigues(rvec_l)
    print(rvec_l, tvec_l)

    _, rvec_r, tvec_r = cv2.solvePnP(objp, corners_r, intrinsic_matrix_right, distortion_right)
    rvec_r, _ = cv2.Rodrigues(rvec_r)
    print(rvec_r, tvec_r)

    cv2.imshow('img_left', frame_markers_l)
    cv2.imshow('img_right', frame_markers_r)

    cv2.waitKey(500)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(objp[:, 0], objp[:, 1], objp[:, 2])
# ax.scatter3D(objp)
# fig.savefig('../../output/task_2/Figure ' + str(i) + '.png')
# ax.scatter3D(origin2[0],origin2[1],origin2[2],c='green')
# ax.scatter3D(0.0,0.0,0.0,c='red')
plt.show()
