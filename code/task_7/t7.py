import cv2 as cv
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

str_left = '../../images/task_7/left_{}.png'
str_right = '../../images/task_7/right_{}.png'
df = (str_left.format(1))

i=0                                                                         #Check which pair of images
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
h, w = left.shape
newcameramtx_l, roi_l=cv.getOptimalNewCameraMatrix(mtx_l,dist_l,(w,h),1, (w,h))

mapx_l, mapy_l = cv.initUndistortRectifyMap(mtx_l, dist_l, None, None, (w,h), 5)
dst1 = cv.remap(left, mapx_l, mapy_l, cv.INTER_LINEAR)

#right only
h,  w = right.shape
newcameramtx_r, roi_r=cv.getOptimalNewCameraMatrix(mtx_r,dist_r,(w,h),1, (w,h))

mapx_r, mapy_r = cv.initUndistortRectifyMap(mtx_r, dist_r, None, None, (w,h), 5)
dst2 = cv.remap(right, mapx_r, mapy_r, cv.INTER_LINEAR)

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

cv.imshow('Feature Matching',img3)#,plt.show()
cv.imwrite('../../output/task_7/Feature Matching '+ str(i) + '.png', img3)#, plt.show()
cv.waitKey(1000)
matches = np.asarray(matches)

intrinsic_matrix_left = np.loadtxt('../../parameters/intrinsic_l.csv', delimiter=',')
intrinsic_matrix_right = np.loadtxt('../../parameters/intrinsic_r.csv', delimiter=',')

# print(distortion)

R = np.loadtxt('../../parameters/R.csv', delimiter = ',')
T = np.loadtxt('../../parameters/T.csv', delimiter = ',')

translate = np.zeros([3,1])
rotate = np.identity(3)
proj1 = np.dot(intrinsic_matrix_left,np.concatenate((rotate,translate),axis=1))
R = np.asarray(R)
T = np.asarray(T)
T = np.reshape(T,(3,1))
proj2 = np.dot(intrinsic_matrix_right, np.concatenate((R,T),axis=1))

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

list_kp1 = np.asarray(list_kp1)
list_kp2 = np.asarray(list_kp2)

triangulate = cv.triangulatePoints(proj1,proj2,list_kp1,list_kp2)

triangulate = np.array(triangulate)
# print(triangulate)

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

E,mask = cv.findEssentialMat(list_kp1,list_kp2,cameraMatrix = intrinsic_matrix_left,method = cv.RANSAC,prob = 0.999,threshold = 1.0)

#Plot the inliers

good_points_l = []
good_points_r = []
kp1_n = []
kp2_n = []
des1_n = []
des2_n = []

for k in range(284):
    if mask[k] == 1:
            good_points_l.append(list_kp1[k])
            good_points_r.append(list_kp2[k])
            kp1_n.append(kp1[k])
            kp2_n.append(kp2[k])
            des1_n.append(des1[k])
            des2_n.append(des2[k])

good_points_l=np.asarray(good_points_l)
good_points_r=np.asarray(good_points_r)
des1_n=np.asarray(des1_n)
des2_n=np.asarray(des2_n)

#New key points using inliers
img4 = cv.drawKeypoints(dst1, kp1_n, None, color=(0,255,0), flags=0)
cv.imshow('Feature Left (inliers)', img4)
cv.imwrite('../../output/task_7/Feature Left (inliers) '+ str(i) + '.png', img4)
cv.waitKey(1000)

img5 = cv.drawKeypoints(dst1, kp2_n, None, color=(0,255,0), flags=0)
cv.imshow('Feature Right (inliers)', img5)
cv.imwrite('../../output/task_7/Feature Right (inliers) '+ str(i) + '.png', img5)
cv.waitKey(1000)

# creating BFMatcher object
bf_n = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Matching the descriptors
matches_n = bf_n.match(des1_n,des2_n)

# Sorting them in the order of their distance
matches_n = sorted(matches_n, key = lambda x:x.distance)

img6 = cv.drawMatches(img4,kp1_n,img5,kp2_n,matches_n[:50],None,flags=2)
cv.imshow('Feature Matching (inliers)', img6)
cv.imwrite('../../output/task_7/Feature Matching (inliers) '+ str(i) + '.png', img6)
cv.waitKey(1000)

triangulate = cv.triangulatePoints(proj1,proj2,good_points_l,good_points_r)

triangulate = np.array(triangulate)
# print(triangulate)

cv.destroyAllWindows()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig2 = plt.figure()
ax=plt.axes(projection='3d')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.scatter(triangulate[0]/triangulate[3],triangulate[1]/triangulate[3],triangulate[2]/triangulate[3])
#fig2.savefig('../../output/task_7/Plot (inliers) ' + str(i) + '.png')


#points, R, t, mask = cv.recoverPose(E, list_kp1, list_kp2)
points, R, t, mask = cv.recoverPose(E, good_points_l, good_points_r)

v = np.array([[-0.25, -0.25, 1], [0.25, -0.25, 1], [0.25, 0.25, 1], [-0.25, 0.25, 1], [0, 0, 0]])
ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])
verts = [[v[0], v[1], v[4]], [v[0], v[3], v[4]],
         [v[2], v[1], v[4]], [v[2], v[3], v[4]], [v[0], v[1], v[2], v[3]]]

# plot sides
ax.add_collection3d(Poly3DCollection(verts,
                                     linewidths=1, edgecolors='r', alpha=.25))

res1 = np.matmul(R, v.T)
f_res = res1 + (10*t)
v = f_res.T
print(v)

ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])

# generate list of sides' polygons of our pyramid
verts = [[v[0], v[1], v[4]], [v[0], v[3], v[4]],
         [v[2], v[1], v[4]], [v[2], v[3], v[4]], [v[0], v[1], v[2], v[3]]]

# plot sides
ax.add_collection3d(Poly3DCollection(verts,
                                     linewidths=1, edgecolors='r', alpha=.25))

print("R = ")
print(R)
print("t = ")
print(t)
plt.show()