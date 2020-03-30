import cv2
import numpy as np
from cv2 import aruco
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np



objp = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)

print(objp.shape)

v = np.array([[-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2], [0, 0, 0]])
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.scatter3D(objp[1:, 0], objp[1:, 1], objp[1:, 2])
#ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])
ax.scatter3D(0.0, 0.0, 0.0, c='red')

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
    rvec_l,_ = cv2.Rodrigues(rvec_l)
    #tvec_l, _ = cv2.Rodrigues(tvec_l)
    rvec_l = np.asarray(rvec_l)
    tvec_l = np.asarray(tvec_l)
    rvec_l = rvec_l.T

    tvec_l = np.matmul(-rvec_l,tvec_l)

    print(5*tvec_l)
    print(rvec_l)

    _, rvec_r, tvec_r = cv2.solvePnP(objp, corners_r, intrinsic_matrix_right, distortion_right)
    rvec_r, _ = cv2.Rodrigues(rvec_r)
    #tvec_r, _ = cv2.Rodrigues(tvec_r)
    #print(rvec_r.shape, tvec_r.shape)

    # vertices of a pyramid

    #ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])
    #
    # # generate list of sides' polygons of our pyramid
    # verts = [[v[0], v[1], v[4]], [v[0], v[3], v[4]],
    #          [v[2], v[1], v[4]], [v[2], v[3], v[4]], [v[0], v[1], v[2], v[3]]]
    #
    # # plot sides
    # ax.add_collection3d(Poly3DCollection(verts,
    #                                      linewidths=1, edgecolors='r', alpha=.25))

    res1 = np.matmul(rvec_l,v.T)
    f_res = res1+(tvec_l)
    v = f_res.T
    #v = np.matmul(v, tvec_l)

    # v = -rvec_l.T * tvec_l
    #print(v)

    #T = np.concatenate(rvec_l,tvec_l)




    ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])

    # generate list of sides' polygons of our pyramid
    verts = [[v[0], v[1], v[4]], [v[0], v[3], v[4]],
             [v[2], v[1], v[4]], [v[2], v[3], v[4]], [v[0], v[1], v[2], v[3]]]

    # plot sides
    ax.add_collection3d(Poly3DCollection(verts,
                                         linewidths=1, edgecolors='r', alpha=.25))


    cv2.imshow('img_left', frame_markers_l)
    cv2.imshow('img_right', frame_markers_r)

    cv2.waitKey(100)

plt.show()


