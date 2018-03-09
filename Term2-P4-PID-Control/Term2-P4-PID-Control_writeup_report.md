# **PID-Control Project**

### Writeup - Submitted by Deepak Mane

---
## Project Outline:

The goals / steps of this project are the following:

In this project two PID controllers were used to control each steering and throttle of a car in a driving simulator. The car drives successfully around the track. The speed was set to 40mph.

The PID controller was implemented in PID.cpp. The method Init initializes the PID with the parameters passed as argument, while the method UpdateError keeps track of the proportional, differential and integral errors of the PID controller (more on that later). Finally the method TotalError() returns the control input as a result (combining P, I and D corrections).

The main program main.cpp handles the communication via uWebSockets to the simulator. We pass the Cross Track Error to the PID controlling the steering, which is used to update/compute de PID error and return the steering value. The same is done with the PID controlling the throttle, but in this case we give the difference of the current speed of the car and the reference speed (which is set to 40mph in our case).

Overview of PID Controllers
PID controllers are a rather basic, but commonly used, class of controllers. PID stands for Proportional, Integral and Derivative Terms. The PID-Controller can be further subdivided into the P-Controller, I-Controller and D-Controller, which can be used separately or as a group to achieve the specific goals depending on the target system.

The P-Controller outputs a correction based uniquely on the amplitude/strength of the input signal. In our case we feed the P-Controller with the Cross Track Error (or the difference between current speed and reference speed). The larger the input signal the more aggresively the P-Controller tries to compensate. The main problem of the P-Controller is that it tends to overshoot (see image below). Combining the P-Controller with the D-Controller can help solve the overshooting problem.

The D-Controller reacts to fast changing input signals. The higher the derivative of the input signal the higher the output of the D-Controller. In the figure below we can see that the PD-Controller overcomes the overshooting problem.

The I-Controller outputs the integral of the error. This is useful to take systemic bias into account (e.g. desalignment of the tyres). In the figure below the PD-Controller shows the best result, since there is no systemic bias in the system, but when systemic bias is present, which is often the case in non-ideal systems, the PID-Controller is the best option.


---
## Rubric Points
### Here I will consider the [Rubric](https://review.udacity.com/#!/rubrics/824/view) Points individually and describe how I addressed each point in my implementation.  

---
### [ 1.] Compilation

CRITERIA : Code compiles without errors with cmake and make.
Given that we've made CMakeLists.txt as general as possible, it's recommend that you do not change it unless you can guarantee that your changes will still compile on any platform.

#### Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is reference template writeup used from other project as a starting point.  

SPECIFICATION :  My project includes the following files:
* `Term2-P4-PID-Control_writeup_report.md`:- This is the writeup which is submitted for this project.
* Code for Project : Files PID.cpp,PID.h,main.cpp.
* Ouput video with vehicles detected - Term2-P4-PID.gif

### [ 2.] Implementation

CRITERIA
The PID procedure follows what was taught in the lessons.

### [ 3.] Reflection

CRITERIA
### Describe the effect each of the P, I, D components had in your implementation.

- **Proportional component (P)**: Makes the car steer towards the _CTE_. It is achieved by multiplying the _CTE_ by _P_ values. It will never reach the _CTE_ but  will oscillate around it, resulting in an unsecure behaviour. That's the reason to use _I_ and _D_ temrs.

- **Integral component (I)**: Compensates the systematic bias. When the car steering has been incorrectly fixed, lets say to the left, the _I_ term should compensate it. This is achieved acumulating the value of the surface between the car position and the _CTE_ over time, and multipying this value by the _I_ coefficient.

- **Differential component (D)**: Counter-steers when the car begins to steer toward the _CTE_. It goes smaller as the car drives towards the _CTE_, avoiding _P_ component tendency to ring and overshoot the center line.

### Describe how the final hyperparameters were chosen.

All the parameters has been chosen by the old _try/error_ technique. I've found that the _I_ component was nearly innocuous in my solution, I've setted it to `0` and obtained better results.

The final values are:

- _P_ = `0.18`
- _I_ = `0`
- _D_ = `3.0`

### [ 4.] Simulation

CRITERIA
The vehicle must successfully drive a lap around the track.
No tire leaves the drivable portion of the track surface. The car does not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle).
You can observe the result in this [images](https://github.com/deepak-mane/SDCND/blob/master/Term2-P4-PID-Control/images/Term2-P4-PID.gif).

---
END OF WRITEUP
---

