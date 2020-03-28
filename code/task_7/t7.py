import cv2 as cv
import numpy as np

str_left = '../../images/task_7/left_{}.png'
str_right = '../../images/task_7/right_{}.png'
df = (str_left.format(1))

i=0                                                                                 #Done for image 0 and 4 in report
left = cv.imread(str_left.format(i), cv.IMREAD_GRAYSCALE)
right = cv.imread(str_right.format(i), cv.IMREAD_GRAYSCALE)

cv.imshow('window_left', left)
cv.imshow('window_right', right)
cv.waitKey(1000)
cv.destroyAllWindows()

#left only
mtx_l = np.loadtxt('../../parameters/intrinsic_l.csv', delimiter=',')
dist_l = np.loadtxt('../../parameters/distortion_l.csv', delimiter=',')
mtx_r = np.loadtxt('../../parameters/intrinsic_r.csv', delimiter=',')
dist_r = np.loadtxt('../../parameters/distortion_r.csv', delimiter=',')
h,  w = left.shape
newcameramtx_l, roi_l=cv.getOptimalNewCameraMatrix(mtx_l,dist_l,(w,h),1, (w,h))

# print(mtx_l)
# print(roi_l)

mapx_l, mapy_l = cv.initUndistortRectifyMap(mtx_l, dist_l, None, None, (w,h), 5)
dst1 = cv.remap(left, mapx_l, mapy_l, cv.INTER_LINEAR)
# x,y,w,h = roi_l
# dst1 = dst1[y:y+h, x:x+w]
# print(dst.shape)


#right only
h,  w = right.shape
newcameramtx_r, roi_r=cv.getOptimalNewCameraMatrix(mtx_r,dist_r,(w,h),1, (w,h))

#right = cv.imread(str_right.format(0))
mapx_r, mapy_r = cv.initUndistortRectifyMap(mtx_r, dist_r, None, None, (w,h), 5)
dst2 = cv.remap(right, mapx_r, mapy_r, cv.INTER_LINEAR)
# x,y,w,h = roi_r
# dst2 = dst2[y:y+h, x:x+w]
# print(dst.shape)

#create orb class object for Left
orb1 = cv.ORB_create()
kp1 = orb1.detect(dst1,None)
kp1, des1 = orb1.compute(dst1, kp1)
img1 = cv.drawKeypoints(dst1, kp1, None, color=(0,255,0), flags=0)
cv.imshow('Feature Left', img1)
cv.imwrite('../../output/task_7/Feature Left '+ str(i) + '.png', img1)
cv.waitKey(1000)

#create orb class object for Right
orb2 = cv.ORB_create()
kp2 = orb2.detect(dst2,None)
kp2, des2 = orb2.compute(dst2, kp2)
img2 = cv.drawKeypoints(dst2, kp2, None, color=(0,255,0), flags=0)
cv.imshow('Feature Right', img2)
cv.imwrite('../../output/task_7/Feature Right '+ str(i) + '.png', img2)
cv.waitKey(1000)

des1 = np.array(des1)
des2 = np.array(des2)

# creating BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Matching the descriptors
matches = bf.match(des1,des2)

# Sorting them in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 15 matches
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:50],None,flags=2)
# print(kp1)
print('@##########################################################################')

cv.imshow('Feature Matching',img3)#,plt.show()
cv.imwrite('../../output/task_7/Feature Matching '+ str(i) + '.png', img3)#, plt.show()
cv.waitKey(1000)
#print(len(matches),len[0](matches))
matches = np.asarray(matches)
print(matches.shape)

intrinsic_matrix_left = np.loadtxt('../../parameters/intrinsic_l.csv', delimiter=',')
intrinsic_matrix_right = np.loadtxt('../../parameters/intrinsic_r.csv', delimiter=',')

# print(distortion)

R = np.loadtxt('../../parameters/R.csv', delimiter = ',')
T = np.loadtxt('../../parameters/T.csv', delimiter = ',')
# F = np.loadtxt('../../parameters/F.csv', delimiter = ',')
# E = np.loadtxt('../../parameters/E.csv', delimiter = ',')

translate = np.zeros([3,1])
rotate = np.identity(3)
proj1 = np.dot(intrinsic_matrix_left,np.concatenate((rotate,translate),axis=1))
R = np.asarray(R)
T = np.asarray(T)
T = np.reshape(T,(3,1))
#print(R.shape,T.shape)
proj2 = np.dot(intrinsic_matrix_right, np.concatenate((R,T),axis=1))

# Tx = np.asscalar(np.array(T[0]))
# Ty = np.asscalar(np.array(T[1]))
# Tz = np.asscalar(np.array(T[2]))
#
# Ess = [[0 for x in range(3)] for y in range(3)]
# Ess[0][0] = 0
# Ess[0][1] = -Tz
# Ess[0][2] = Ty
# Ess[1][0] = Tz
# Ess[1][1] = 0
# Ess[1][2] = -Tx
# Ess[2][0] = -Ty
# Ess[2][1] = Tx
# Ess[2][2] = 0
#
# #[[0,-Tz,Ty],[Tz, 0 , -Tx],[-Ty, Tx, 0]]
# Ess = np.asarray(Ess)
# #Ess = np.dot(Ess,R)
# print(Ess)
# print(Ess.shape)
# print('EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
#
# list_kp1 = []
# list_kp2 = []
# list_kp1_hom = []
# list_kp2_hom = []
#
# # For each match...
# for mat in matches:
#
#     # Get the matching keypoints for each of the images
#     img1_idx = mat.queryIdx
#     img2_idx = mat.trainIdx
#
#     # x - columns
#     # y - rows
#     # Get the coordinates
#     (x1, y1) = kp1[img1_idx].pt
#     (x2, y2) = kp2[img2_idx].pt
#
#     # Append to each list
#     list_kp1.append([(x1, y1)])
#     list_kp2.append([(x2, y2)])
#
#     #Same in homogeneous
#     list_kp1_hom.append([(x1, y1, 0)])
#     list_kp2_hom.append([(x2, y2, 0)])
#
#
# #print(list_kp1)
# list_kp1 = np.asarray(list_kp1)
# list_kp2 = np.asarray(list_kp2)
# list_kp1_hom = np.asarray(list_kp1_hom)
# list_kp2_hom = np.asarray(list_kp2_hom)
#
#
# good_points = []
#
# print(list_kp1_hom[0].shape, list_kp2_hom[0].shape, Ess.shape)
#
# #print(good_points)
# good_p = []
# for i in range(203):
#     const = np.dot(list_kp1_hom[i],Ess)
#     good_points.append(float(np.dot(const,np.transpose(list_kp1_hom[i]))))
# print(good_points)
# index=[]
# for el in range(203):
#     if(good_points[el]==0.0):
#         good_p.append(good_points[el])
#         index.append(el)
#
# print(good_p)
# print(len(good_p))
#
# l_kp1 = list_kp1[index]
# l_kp2 = list_kp2[index]
# # print(len(l_kp1))
# # print(l_kp1)
# kp1_n = [kp1[x] for x in index]
# kp2_n = [kp2[x] for x in index]
# print(kp1_n)
#
# img2_n1 = cv.drawKeypoints(dst2, kp1_n, None, color=(0,255,0), flags=0)
# img2_n2 = cv.drawKeypoints(dst2, kp2_n, None, color=(0,255,0), flags=0)
#
# cv.imshow('Feature Matching with good points 1',img2_n1)#,plt.show()
# cv.imwrite('../../output/task_3/Feature Match with good points 1.png', img2_n1)
# cv.imshow('Feature Matching with good points 2',img2_n2)#,plt.show()
# cv.imwrite('../../output/task_3/Feature Match with good points 2.png', img2_n2)

#cv.imwrite('../../output/task_3/Feature Matching with reduced points '+ str(i) + '.png', img3)#, plt.show()

kp1 = np.asarray(kp1)
kp2 = np.asarray(kp2)

list_kp1 = []
list_kp2 = []
list_kp1_hom = []
list_kp2_hom = []

# For each match...
for mat in matches:

    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    # x - columns
    # y - rows
    # Get the coordinates
    (x1, y1) = kp1[img1_idx].pt
    (x2, y2) = kp2[img2_idx].pt

    # Append to each list
    list_kp1.append([(x1, y1)])
    list_kp2.append([(x2, y2)])

    #Same in homogeneous
    list_kp1_hom.append([(x1, y1, 0)])
    list_kp2_hom.append([(x2, y2, 0)])

#print(list_kp1_hom)
list_kp1 = np.asarray(list_kp1)
list_kp2 = np.asarray(list_kp2)
#print(list_kp1)
#cv.waitKey(1000)

triangulate = cv.triangulatePoints(proj1,proj2,list_kp1,list_kp2)

triangulate = np.array(triangulate)
print(triangulate)

cv.destroyAllWindows()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax=plt.axes(projection='3d')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.scatter(triangulate[0]/triangulate[3],triangulate[1]/triangulate[3],triangulate[2]/triangulate[3])
# cv.imwrite('../../output/task_3/Plot '+ str(i) + '.png', fig)
fig.savefig('../../output/task_7/Plot '+ str(i) + '.png')
plt.show()

#retval,mask = cv.findEssentialMat(list_kp1,list_kp2,cameraMatrix = intrinsic_matrix_left,method = 'RANSAC',prob = 0.999,threshold = 1.0,mask = [] )
E,mask = cv.findEssentialMat(list_kp1,list_kp2,cameraMatrix = intrinsic_matrix_left,method = cv.RANSAC,prob = 0.999,threshold = 1.0)
points, R, t, mask = cv.recoverPose(E, list_kp1, list_kp2)
print("R = ")
print(R)
print("t = ")
print(t)
