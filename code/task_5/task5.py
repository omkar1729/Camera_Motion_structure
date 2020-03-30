import cv2
import numpy as np
from math import cos, sin, radians

objp = np.zeros((6*9,2), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# print(objp)

for i in objp:
    i[0] = 300 + 10*i[0]
    i[1] = 800 + 10*i[1]
# print(objp)

object_points = []
image_points_left = []
image_points_right = []
shape = ()

intrinsic_matrix_left = np.loadtxt('../../parameters/intrinsic_l.csv', delimiter=',')
intrinsic_matrix_right = np.loadtxt('../../parameters/intrinsic_r.csv', delimiter=',')
distortion_left = np.loadtxt('../../parameters/distortion_l.csv',delimiter=',')
distortion_right = np.loadtxt('../../parameters/distortion_r.csv',delimiter=',')

for i in range(1):
    left_img = cv2.imread('../../images/task_5/left_'+str(i)+'.png')
    right_img = cv2.imread('../../images/task_5/right_' + str(i) + '.png')
    h,w = left_img.shape[:-1]
    shape = (w,h)
    # print(shape)
    R = np.identity(3)
    undistort_map_left_x, undistort_map_left_y = cv2.initUndistortRectifyMap(intrinsic_matrix_left, distortion_left, R,
                                                                             intrinsic_matrix_left, shape, cv2.CV_32FC1)
    undistort_map_right_x, undistort_map_right_y = cv2.initUndistortRectifyMap(intrinsic_matrix_right, distortion_right,
                                                                               R, intrinsic_matrix_right, shape,
                                                                               cv2.CV_32FC1)

    remap_left_img = cv2.remap(left_img, undistort_map_left_x,undistort_map_left_y, cv2.INTER_LINEAR)
    remap_right_img = cv2.remap(right_img, undistort_map_right_x,undistort_map_right_y, cv2.INTER_LINEAR)

    ret_l, corner_left = cv2.findChessboardCorners(remap_left_img, (9, 6))
    ret_r, corner_right = cv2.findChessboardCorners(remap_right_img, (9, 6))

    result_left = cv2.drawChessboardCorners(remap_left_img, (9, 6), corner_left, ret_l)
    result_right = cv2.drawChessboardCorners(remap_right_img, (9, 6), corner_right, ret_r)

    image_points_left.append(corner_left)
    image_points_right.append(corner_right)

    object_points.append(objp)

    cv2.imshow('img_left', result_left)
    cv2.imshow('img_right', remap_right_img)
    cv2.waitKey()

cv2.destroyAllWindows()

# print(corner_left.shape)

image_points_left = np.float32(image_points_left).reshape(-1,1,2)
image_points_right = np.float32(image_points_right).reshape(-1,1,2)
object_points = np.float32(object_points).reshape(-1,1,2)

# print(image_points_left)
# print(np.shape(object_points))

homography_matrix_left, mask = cv2.findHomography(image_points_left,object_points,0)
homography_matrix_right, mask = cv2.findHomography(image_points_right,object_points,0)
print(homography_matrix_left)


im_dst_left = cv2.warpPerspective(remap_left_img, homography_matrix_left, (shape[0]*2, shape[1]*2))
im_dst_right = cv2.warpPerspective(remap_right_img, homography_matrix_right, (shape[0]*2, shape[1]*2))
cv2.namedWindow('left_warped', cv2.WINDOW_NORMAL)
cv2.namedWindow('right_warped', cv2.WINDOW_NORMAL)
cv2.imshow('left_warped', im_dst_left)
cv2.imwrite('../../output/task_5/left_warp ' + str(i) + '.png',im_dst_left)
cv2.imshow('right_warped', im_dst_right)
cv2.imwrite('../../output/task_5/right_warp ' + str(i) + '.png',im_dst_right)

cv2.waitKey()
