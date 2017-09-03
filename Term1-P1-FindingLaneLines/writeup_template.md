**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_files folder Contains source Images as 

solidWhiteCurve.jpg
solidYellowCurve.jpg
solidYellowLeft.jpg
solidWhiteRight.jpg
solidYellowCurve2.jpg
whiteCarLaneSwitch.jpg
---
[//]: # (Video References)
challenge.mp4
solidWhiteRight.mp4
solidYellowLeft.mp4

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 11 steps. 
All the steps are listed with and the input taken to accomplish that step is mentioned in [square brackets]
The sequence is followed the same in the functions in the program and few steps are clubbed together in particular function to make easy h

PIPELINE :-
STEP1.] CREATED GRAYSCALE IMAGE [FROM SOURCE IMAGE CONVERTED TO GRAYSCALE] 
STEP2.] APPLIED GAUSSIANBLUR [INPUT USED —> GRAYSCALE IMAGE FROM SOURCE IMAGE 
STEP3.] APPLIED CANNY’S EDGE DETECTION ALGORITHM [INPUT USED —> GAUSSIANBLUR APPLIED IMAGE  ON GRAYSCALE IMAGE]
STEP4.] FINDING EDGES OF IMAGE [INPUT USED —> CANNY’s OUTPUT ON GAUSSIANBLUR APPLIED IMAGE]
STEP5.] FINDING VERTICES [FROM SOURCE IMAGE USING "IMSHAPE" AND "NP.ARRAY" MODULE]
STEP6.] MASKING OF IMAGE [STARTED WITH BLANK MASK AND FILLED THE MASK WITH PIXELS DEFINED BY VERTICES USING MODULE CV2.FILLPOLY]
STEP7.] APPLYING BITWISE OPERATION WITH AND LOGIC [USING MODULE "CV2.BITWISE_AND" ON IMAGE]
STEP8.] REGION OF INTEREST [INPUT USED MASKED IMAGE DERIVED FROM ABOVE OPERATIONS OF EDGES,VERTICES,MASKING AND BITWISE]
STEP9.] DRAIWING LINES BY APPLYING HOUGH LINES ALGORITHM TO MASKED EDGES [INPUT USED —> RHO,THETA,THRESHOLD,MIN_LINE_LENGTH AND MAX_LINE_GAP ]
STEP10.] USING ADDWEIGHTED MODULE TO THE LINE IMAGE TO FIND THE LINE EDGES
STEP11.] APPLYING STEPS FROM CREATING GRAYSCALE IMAGE TILL FINDING THE LINE EDGES TO VEDIO FILES USING clip = clip.fl_image 

======
Images are Included in the test_files folder and the Output Image files are Source File name with Prefix of "output_"
./test_files folder Contains Output Images as 
solidYellowCurve2_output.png
whiteCarLaneSwitch_output.png
solidWhiteCurve_output.png
solidYellowCurve_output.png
solidWhiteRight_output.png
solidYellowLeft_output.png

Output Videos in test_files folder is suffixed with "output_" to the original videos
./test_files folder Contains Output Videos as 
output_solidWhiteRight.mp4
output_solidYellowLeft.mp4
output_challenge.mp4


### 2. Identify potential shortcomings with your current pipeline

The Lines drawn seem to be little not on the exact SoldYellowCurve.


### 3. Suggest possible improvements to your pipeline

None is being realised at present