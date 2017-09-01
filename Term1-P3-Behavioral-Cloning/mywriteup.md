# **Behavioral Cloning Project**

### Writeup - Submitted by Deepak Mane

---

The goals / steps taken to completed this Project are as follows:
* Used the simulator to collect data of good driving behavior [ Provided by Udacity from [here] ](https://github.com/udacity/self-driving-car-sim)
* Saved the Data on Local system under ~/Desktop/data
* Validated the mimium dependences (CarND Term1 Starter Kit as reference) for running the Project like (Python3, TensorFlow, Keras, Numpy, OpenCV)
* Built, a convolution neural network in Keras that predicts steering angles from images
* Trained and validated the model with a training and validation set on both the Tracks
* Tested that the model successfully drives around track one without leaving the road. Attempted to test the car to Run on track two but it was not successful. Posted below few findings about it.
* Summarized the results with a written report as Below


[//]: # (Image References)

[image1]: ./writeup_images/00_original.png "Normal Image"
[image2]: ./writeup_images/01_translate.png "Grayscaling"
[image3]: ./writeup_images/02_curvature.png "Recovery Image"
[image4]: ./writeup_images/03_incline_down.pn "Recovery Image"
[image5]: ./writeup_images/04_incline_up.png "Recovery Image"
[image6]: ./writeup_images/05_flip.png "Flipped Image"
[image7]: ./writeup_images/placeholder_small.png 


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Required Files

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
[Are all required files submitted?]
SPECIFICATION : My project includes the following files:
* model.py :- This is the script created to train the model
* drive.py :- This script is as used without any changes for driving the car in autonomous mode
* model.h5 :- This file has the data which represents trained convolution neural network 
* writeup_report.md :- Summarizes the results

### Quality of Code

* Is the code functional?	

SPECIFICATION : The model provided can be used to successfully operate the simulation.

#### 2. Submission includes functional code

* Is the code usable and readable?	The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed.

SPECIFICATION : 
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

* Has an appropriate model architecture been employed for the task?	The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model.
* Has an attempt been made to reduce overfitting of the model?	Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting.
* Have the model parameters been tuned appropriately?	Learning rate parameters are chosen with explanation, or an Adam optimizer is used.
* Is the training data chosen appropriately?	Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track).

SPECIFICATION : 

### Architecture and Training Documentation

* Is the solution design documented?	The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.
* Is the model architecture documented?	The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.
* Is the creation of the training dataset and training process documented?	The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset must be included.

SPECIFICATION : 

