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
[image1]: ./outout_images/image1.png
[image2]: ./outout_images/image2.png
[image3]: ./outout_images/image3.png
[image4]: ./outout_images/image4.png
[image5]: ./outout_images/image5.png
[image6]: ./outout_images/image6.png
[image7]: ./outout_images/image7.png
[image8]: ./outout_images/image8.png
[image9]: ./outout_images/image9.png
[image10]: ./outout_images/image10.png
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

![alt text][image1]

The dataset used contained 2826 cars and 8968 not car images. This dataset is unbalanced. I decided to leave it unbalanced since in the project video not car images far exceed the car images. The code for this step is contained in the code cell 2 of the IPython notebook.

I started by exploring the color features - spatial binning and color histogram. For spatial binning, I reduced the image size to 16,16 and the plot below shows the difference in spatial binning features between car and notcar images for channel - RGB. The plot delta shows the difference b/w car and notcar spatial binned features

The code for this step is contained in the code cell 3 and 6 of the IPython notebook. In the end I decided to not use color features (histogram and spatial binning) as it adversely affected performance.

Next I looked at HOG features using skimage.hog() functions. The key parameters are 'orientations', 'pixels_per_cell' and 'cells_per_block'. The num of orientations is the number of gradient directions. The pixels_per_cell parameter specifies the cell size over which each gradient histogram is computed. The cells_per_block parameter specifies the local area over which the histogram counts in a given cell will be normalized. To get a feel for the affect of pixels_per_cell and cells_per_block, I looked at hog images with different settings for pixels per cell and cells per block. All the images below are from gray scale. The code for this step is contained in the code cell 4 of the IPython notebook.


![alt text][image2]

---
* Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally settled on the choice:

color_space = 'YCrCb' - YCrCb resulted in far better performance than RGB, HSV and HLS
orient = 9 # HOG orientations - I tried 6,9 and 12. Model performance didn't vary much
pix_per_cell = 16 - I tried 8 and 16 and finally chose 16 since it signficantly decreased computation time
cell_per_block = 1 - I tried 1 and 2. The performance difference b/w them wasn't much but 1 cell per block had significantly less no. of features and speeded up training and pipeline
hog_channel = 'ALL' - ALL resulted in far better performance than any other individual channel
I spent a lot of time narrowing down on these parameters. In the beginning I relied on the test accuracy in SVM classifier to choose parameters but then found that most combinations had very high accuracy (b/w 96% and 98%) and this wasn't indicative of performance in the video. So these parameters were chosen after painstakingly trying and observing performance in the video. The code for this step is contained in the code cell 9 and 10 of the IPython notebook.

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

![alt text][image3] (./output_images/image3.png)
In the sliding window technique, for each window we extract features for that window, scale extracted features to be fed to the classifier, predict whether the window contains a car using our trained Linear SVM classifier and save the window if the classifier predicts there is a car in that window.

For the final model I chose 2 window sizes - [(96,96), (128,128)] and correspoding y_start_stop of [[390, 650], [390, None]]. I found that the performance was improved with x_start_stop=[700, None] since it reduced the search area to the right side lanes. I chose an overlap of 0.7

---
* Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some examples of test images from my classifier. As you can see there are multiple detections and false positives. To smoothen out multiple detections and to remove false positives, I used the technique for generating heatmaps that was suggested in the lectures and set a threshold of 2.
The code for this step is in cell 14 of the IPython notebook.

![alt text][image4] (./output_images/image4.png)
![alt text][image5] (./output_images/image5.png)
![alt text][image6] (./output_images/image6.png)
![alt text][image7] (./output_images/image7.png)
![alt text][image8] (./output_images/image8.png)
![alt text][image9] (./output_images/image9.png)

---

### [ 4.] Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

SPECIFICATION :  Below is the Link in this writeup for the final output video submitted with the project.

[Video output](./vehicle_detection.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. I combined detection over 20 frames (or using the number of frames available if there have been fewer than 20 frames before the current frame). From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I found best performance with threshold parameter of 22. I used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

The code for this is in the vehicle detection pipeline is in cell 15 and 16

Here's an example result showing the heatmap from the last 20 frames of video, the result of scipy.ndimage.measurements.label() on the heatmap and the bounding boxes then overlaid on the last frame of video:

* Here are six frames and their corresponding heatmaps:

![alt text][image4] (./output_images/image4.png)
![alt text][image5] (./output_images/image5.png)
![alt text][image6] (./output_images/image6.png)
![alt text][image7] (./output_images/image7.png)
![alt text][image8] (./output_images/image8.png)
![alt text][image9] (./output_images/image9.png)

* Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image10] (./output_images/image10.png)



---

### [ 5.] Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Two problems that I faced were:

I found that the test accuracy in the classifier was not a good predictor of actual performance in the video. Most model combinations had an accuracy of 97%+ but only a few had good performance in the video. This was a bit surprising. I think this is because I didn't put in extra work in making sure that examples in training and testing were distinct. As a result the model overfit to the training data. To identify the best model, I tested performance in the video.

Once the video pipeline was working, it was detection false positives in some frames and not detecting the car in other frames. Careful tuning of num of frames over which windows are added and thresholding parameter were needed. Ideally there should be a way of modifying these parameters for different sections of the video.

My biggest concern with the approach here is that it relies heavily on tuning the parameters for window size, scale, hog parameters, threshold etc. and those can be camera/track specific. I am afraid that this approach will not be able to generalize to a wide range of situations. And hence I am not very convinced that it can be used in practice for autonomously driving a car.

Here are a few other situations where the pipeline might fail:

I am not sure this model would perform well when it is a heavy traffic situations and there are multiple vehicles. You need something with near perfect accuracy to avoid bumping into other cars or to ensure there are no crashes on a crossing.

The model was slow to run. It took 6-7 minutes to process 1 minute of video. I am not sure this model would work in a real life situation with cars and pedestrians on thr road.

To make the code more robust we can should try the following:

Reduce the effect of time series in training test split so that the model doesn't overfit to training data

Instead of searching for cars in each image independently, we can try and record their last position and search in a specific x,y range only

Modify HOG to extract features for the entire image only once.

