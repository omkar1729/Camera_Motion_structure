# Perception_project_2b
Sarthake Report - 

Task 5 requires us to transform points from the given image (i.e. image plane) to world plane. 
This is done using 2d-2d point correspondences. First, a camera’s image of the chessboard is loaded. The camera parameters, that were found in the previous tasks, are loaded from the csv files found in the parameter folder. Once these parameters are uploaded, the image loaded are made to go through undistortion using initundistortrectify() function. This function creates two maps that helps obtain the values of pixels coordinates of the new remapped image. This remapping is done using the remap() function. This function simply calculates the pixel values of the new remaped image using the furmula: 
dst(x,y) = src(map_x(x,y),map_y(x,y))

Then ‘findchessboardcorners’ function is called to get the image coordinates of chessboard corners. Along with it, we create the 2d real world points of the chessboard corners. It is given in the problem statement that the coordinates of the real world are scaled by a factor of 10 and translated to (300,800). Using this information we know the coordinates of chessboard corners in the destination plane, i.e. real world plane are of the form (300,800); (310,800)....(380,850). These points are created and stored in a numpy array. 
Then the function findHomography is called by inputting the image coordinates and real world 2d coordniates that we have stored to create the homography matrix. The homography matrix obtained for image left_0 is shown below:

[[ 2.81045250e-01  2.10290737e+00  1.09099094e+02]
 [-1.64597664e+00  6.63394663e+00  4.56803259e+02]
 [-1.85774073e-03  6.25159222e-03  1.00000000e+00]

After we obtain the homography matrix, we use it in the function warpPerspective to perform perspective transformation and obtain the desired image as shown in Fig 1.0. Fig1.0 a. Is the warped image of left_0 whereas Fig1.0 b. Is the warped image of the right_0





					a. Warped image of left_0


					b. Warped image of right_0

The warp function helps map one image to another where the target object in the image is projected onto one plane, just like in the real world. This is a form of perspective transformation. It takes in two inputs. First is the source image that is to be transformed. Second is the homography matrix or the transformation matrix that will apply the required transformation to the source image. It also provides the user the option to input the size of the target image after transformation has been applied. 
