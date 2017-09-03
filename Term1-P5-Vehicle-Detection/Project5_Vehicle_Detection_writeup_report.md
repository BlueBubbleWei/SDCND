# **Vehicle Detection Project**

### Writeup - Submitted by Deepak Mane

---
## Project Outline:

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/image1.png
[image11]: ./output_images/image1.1.png
[image2]: ./output_images/image2.png
[image3]: ./output_images/image3.png
[image4]: ./output_images/image4.png
[image5]: ./output_images/image5.png
[image6]: ./output_images/image6.png
[image7]: ./output_images/image7.png
[image8]: ./output_images/image8.png
[image9]: ./output_images/image9.png
[image10]: ./output_images/image10.png
[video1]: ./vehicle_detection.mp4

---
## Rubric Points
### Here I will consider the [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points individually and describe how I addressed each point in my implementation.  

---
### [ 1.] Writeup / README

* Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

SPECIFICATION :  My project includes the following files:
* `Project5_Vehicle_Detection_writeup_report.md`:- This is the writeup which is submitted for this project.
* Code for Project pipeline is in [Project5_Vehicle_Detection.ipynb](https://github.com/deepak-mane/SDCND/blob/master/Term1-P5-Vehicle-Detection/Project5_Vehicle_Detection.ipynb).
* The images in [test_images](https://github.com/deepak-mane/SDCND/blob/master/Term1-P5-Vehicle-Detection/test_images) are for testing your pipeline on single frames.
* The images in [output_images](https://github.com/deepak-mane/SDCND/blob/master/Term1-P5-Vehicle-Detection/output_images) are from Output from each stage of Pipeline.
* Ouput video with vehicles detected - vehicle_detection.mp4

### [ 2.] Histogram of Oriented Gradients (HOG)

* Explain how (and identify where in your code) you extracted HOG features from the training images.

SPECIFICATION :  I defined the PIPELINE for the project as follows:

STEP1. Load car and not car data from files provided in the resources,Data exploration

STEP2. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images

STEP3. Visualize HOG Features.

STEP4. Visualize Spatial Binning Features and Normalization of color features.

STEP5. Train a HOG Linear SVC Classifier.

STEP6. Sliding window implementation.

STEP7. Test performance of Classifier on sample images.

STEP8. Run Pipeline on a video stream and create a heat map of recurring detections frame by frame.


I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![`vehicle` and `non-vehicle` image][image1]

The dataset used contained 2826 cars and 8968 not car images. This dataset is unbalanced. I decided to leave it unbalanced since in the project video not car images far exceed the car images. The code for this step is contained in the code cell 2 of the IPython notebook.

I started by exploring the color features - spatial binning and color histogram. For spatial binning, I reduced the image size to 16,16 and the plot below shows the difference in spatial binning features between car and notcar images for channel - RGB. The plot delta shows the difference b/w car and notcar spatial binned features

![difference in spatial binning features between car and notcar imagesimage][image11]

The code for this step is contained in the code cell 3 and 6 of the IPython notebook. In the end I decided to not use color features (histogram and spatial binning) as it adversely affected performance.

Next I looked at HOG features using skimage.hog() functions. The key parameters are 'orientations', 'pixels_per_cell' and 'cells_per_block'. The num of orientations is the number of gradient directions. The pixels_per_cell parameter specifies the cell size over which each gradient histogram is computed. The cells_per_block parameter specifies the local area over which the histogram counts in a given cell will be normalized. To get a feel for the affect of pixels_per_cell and cells_per_block, I looked at hog images with different settings for pixels per cell and cells per block. All the images below are from gray scale. The code for this step is contained in the code cell 4 of the IPython notebook.


![alt text][image2]

---
* Explain how you settled on your final choice of HOG parameters.

Below are listed some of paramets which I arrived at after few iterations:

1. color_space = 'YCrCb' - YCrCb resulted in far better performance than RGB, HSV and HLS
2. orient = 9 # HOG orientations - I tried 6,9 and 12. Model performance didn't vary much
3. pix_per_cell = 16 - I tried 8 and 16 and finally chose 16 since it signficantly decreased computation time
4. cell_per_block = 1 - I tried 1 and 2. The performance difference b/w them wasn't much but 1 cell per block had significantly less no. of features and speeded up training and pipeline
5. hog_channel = 'ALL' - ALL resulted in far better performance than any other individual channel

Reasonable amoutn of time was spend to choose these parameters. In the beginning I relied on the test accuracy in SVM classifier to choose parameters but then found that most combinations had very high accuracy (b/w 96% and 98%) and this wasn't indicative of performance in the video. So these parameters were chosen after painstakingly trying and observing performance in the video. The code for this step is contained in the code cell 9 and 10 of the IPython notebook.

---
* Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I followed the steps below for training the classifier

Format features using np.vstack and StandardScaler().
Split data into shuffled training and test sets
Train linear SVM using sklearn.svm.LinearSVC().
The code for this step is in cell 10 of the IPython notebook.

### [ 3.] Sliding Window Search

* Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To implement sliding windows, I narrowed the search area to lower half of the image and searched with different window sizes. Small windows were limited to band 400 pixels to 650 pixels since small cars are more likely to occur farther on the horizon. Below is an example of searching with windows of different sizes.

![Sliding Window Search][image3] 
In the sliding window technique, for each window we extract features for that window, scale extracted features to be fed to the classifier, predict whether the window contains a car using our trained Linear SVM classifier and save the window if the classifier predicts there is a car in that window.

For the final model I chose 2 window sizes - [(96,96), (128,128)] and correspoding y_start_stop of [[390, 650], [390, None]]. I found that the performance was improved with x_start_stop=[700, None] since it reduced the search area to the right side lanes. I chose an overlap of 0.7

---
* Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some examples of test images from my classifier. As you can see there are multiple detections and false positives. To smoothen out multiple detections and to remove false positives, I used the technique for generating heatmaps that was suggested in the lectures and set a threshold of 2.
The code for this step is in cell 14 of the IPython notebook.

![Correct Detection][image4]
![False Detection][image5] 
![False Detection][image6] 
![Correct Detection][image7]
![Correct Detection][image8] 
![Correct Detection][image9] 


### [ 4.] Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

SPECIFICATION :  Below is the Link in this writeup for the final output video submitted with the project.

[Video output][video1]

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. I combined detection over 20 frames (or using the number of frames available if there have been fewer than 20 frames before the current frame). From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I found best performance with threshold parameter of 22. I used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

The code for this is in the vehicle detection pipeline is in cell 15 and 16

Here's an example result showing the heatmap from the last 20 frames of video, the result of scipy.ndimage.measurements.label() on the heatmap and the bounding boxes then overlaid on the last frame of video:

* Here are six frames and their corresponding heatmaps:

![Correct Detection][image4]
![False Detection][image5] 
![False Detection][image6] 
![Correct Detection][image7]
![Correct Detection][image8] 
![Correct Detection][image9] 

* Here the resulting bounding boxes are drawn onto the last frame in the series:
![Last Frame][image10] 


---

### [ 5.] Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The test accuracy from the classifier cannot be considered as a acceptable predictor showing the actual performance in the video. Most model combinations had an Good accuracy but only a few had good performance in the video. This was a bit surprising.I suppose there was supposed to more training required for the classifier. As a result the model overfit to the training data. To identify the best model, I tested performance in the video.

After video pipeline was working, it was detecting false positives in some frames and not detecting the car in other frames. Careful tuning of num of frames over which windows are added and thresholding parameter were needed. Need to find a method modifying these parameters for different sections of the video.

To have fine tuning of the parameters for window size, scale, hog parameters, threshold etc. and those can be camera/track specific needs extra experimentation with different algorithms before having it in actual practice of driving autonomous car.
