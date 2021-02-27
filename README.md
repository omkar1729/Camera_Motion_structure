

# Camera Motion and Structure

**Task 5 -** 

Task 5 requires us to transform points from the given image (i.e. image plane) to world plane.

This is done using 2d-2d point correspondences. First, a camera’s image of the chessboard is loaded. The camera parameters, that were found in the previous tasks, are loaded from the csv files found in the parameter folder. Once these parameters are uploaded, the image loaded are made to go through undistortion using initundistortrectify() function. This function creates two maps that helps obtain the values of pixels coordinates of the new remapped image. This remapping is done using the remap() function. This function simply calculates the pixel values of the new remaped image using the furmula:

dst(x,y) = src(map\_x(x,y),map\_y(x,y))

Then ‘findchessboardcorners’ function is called to get the image coordinates of chessboard corners. Along with it, we create the 2d real world points of the chessboard corners. It is given in the problem statement that the coordinates of the real world are scaled by a factor of 10 and translated to (300,800). Using this information we know the coordinates of chessboard corners in the destination plane, i.e. real world plane are of the form (300,800); (310,800)....(380,850). These points are created and stored in a numpy array.

Then the function findHomography is called by inputting the image coordinates and real world 2d coordniates that we have stored to create the homography matrix. 

After we obtain the homography matrix, we use it in the function warpPerspective to perform perspective transformation and obtain the desired image as shown in Fig 1.0. Fig1.0 a. Is the warped image of left\_0 whereas Fig1.0 b. Is the warped image of the right\_0


| Warped Image Left | Warped Image Right |
| -------------  | ---------------- |
| <img src="https://github.com/omkar1729/Perception_Project_2b/blob/master/output/task_5/left_warp%200.png" width="400" > | <img src="https://github.com/omkar1729/Perception_Project_2b/blob/master/output/task_5/right_warp%200.png" width="400" > |



The warp function helps map one image to another where the target object in the image is projected onto one plane, just like in the real world. This is a form of perspective transformation. It takes in two inputs. First is the source image that is to be transformed. Second is the homography matrix or the transformation matrix that will apply the required transformation to the source image. It also provides the user the option to input the size of the target image after transformation has been applied.

**Task 6**

In this task we calculated the camera pose with help of ArUco markers. First, we detected the ArUco markers so that we get the 2d cordinates on image plane. We know the 3d world cordinates of those points. Hence, we are left with the task of obtaining the camera pose from these 3d – 2d point correspondences. This task is achieved with the help of Perspective-n-point algorithm. This is implemented in openCV as cv2.solvePnP(). The ArUco markers are represented by blue points and top left corner is marked red.

This function is provided with inputs of 3d points, 2d points, camera intrinsic matrix and distortion coefficients. This function outputs a rotation and translation vector. We need to interpret these vectors carefully to obtain the camera pose with respect to world reference frame. The rotation vector needs to be converted to rotation matrix using the function cv2.Rodrigues(). Now the rotation and translation matrix transform the points from world reference frame to camera reference frame. But we want the opposite. Thus, we apply the inverse transformation. Hence, we don’t use the output of solvePnP function directly. 



Below are the camera poses obtained for all (11) left images and right images respectively
| Camera pose left| Camera pose right|
| -------------  | ---------------- |
| <img src="https://github.com/omkar1729/Perception_Project_2b/blob/master/output/task_6/img.png" width="400" > | <img src="https://github.com/omkar1729/Perception_Project_2b/blob/master/output/task_6/img_1.png.png" width="400" > |

**These are the images for detected ArUco markers -**

| <img src="https://github.com/omkar1729/Perception_Project_2b/blob/master/output/task_6/Aruco_detected_left%200.png" width="400" > | <img src="https://github.com/omkar1729/Perception_Project_2b/blob/master/output/task_6/Aruco_detected_right%205.png" width="400" > |

**Task 7 -**

The findEssentialMat() is used to calculate the essential matrix from the corresponding points in the two images, E and the mask array with every element set as 0 for outliers and 1 for inliers. These are the two outputs generated form the function and the parameters are the key points from the two images (points1, points2), the camera matrix which is the same as the camera intrinsic matrix obtained from the calibration results and the method for computing the matrix which is RANSAC in this case. The function also has a probability and threshold parameter. The probability (prob) parameter is used to specify the desirable level of confidence that the estimated matrix is correct, which is taken as 0.999 in the code. The threshold parameter is the maximum distance from a point to an epipolar line in pixels, beyond which the point is considered an outlier and is not used to compute E and has been taken as 1.0.

The recoverPose() is used obtain the relative camera rotation and translation matrices from an estimated essential matrix and the key points in the two images. It also outputs the mask array which marks the inliers for the key points for the E matrix. The function takes the essential matrix E, the two keypoints and the instrinsic camera matrix as input parameters.

Pair 2
| Matching Feature points from the two views in pair 2| Inliers of matched feature points after calculating the essential matrix E for pair 2|
| -------------  | ---------------- |
| <img src="https://github.com/omkar1729/Perception_Project_2b/blob/master/output/task_7/Feature%20Matching%201.png" width="800" > | <img src="https://github.com/omkar1729/Perception_Project_2b/blob/master/output/task_7/Feature%20Matching%20(inliers)%201.png" width="800" > |





Reconstructed 3D points and camera pose for pair 2

<p align="center">
  <img src="https://github.com/omkar1729/Perception_Project_2b/blob/master/output/task_7/Plot%20(inliers)%201.png" width="600" >
</p>

The essential, rotation and translation matrices obtained form the code are given below.


E =

\[\[ 0.00201718 -0.64367885 0.11147717\]

 \[ 0.61853338 -0.00498932 -0.3222911 \]

 \[-0.1127296 0.27189932 0.01136274\]\]

R =

\[\[ 0.99659025 -0.01103807 -0.08176812\]

 \[ 0.01163722 0.99990879 0.0068545 \]

 \[ 0.081685 -0.00778268 0.99662781\]\]

t =

\[\[-0.38274353\]

 \[-0.16443849\]

 \[-0.90910251\]\]



**Requirements for running the code -**

1) The code is tested on opencv 3.2.0. The code is not stable on openCV 4.2.0 as it crashes only for 3<sup>rd</sup> image for task 6.

2) Python3

3) from mpl\_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

