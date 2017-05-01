**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center.jpg "Center"
[image3]: ./examples/rec1.jpg "Recovery Image"
[image4]: ./examples/rec2.jpg "Recovery Image"
[image5]: ./examples/rec3.jpg "Recovery Image"
[image51]: ./examples/rec4.jpg "Recovery Image"
[image6]: ./examples/normal.png "Normal Image"
[image7]: ./examples/flipped.png "Flipped Image"
[image8]: ./examples/progress.png "Progress Image"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
**Files Submitted & Code Quality**

1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

3. Submission code is usable and readable

The model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Model Architecture and Training Strategy

1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.ipynb lines 80-84)

The model includes RELU layers to introduce nonlinearity (code lines 80-94), and the data is normalized in the model using a Keras lambda layer (code line 78).

2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.ipynb line 94).

4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I made 2 circles in bouth directions. 

For details about how I created the training data, see the next section. 

***Model Architecture and Training Strategy***

**1. Solution Design Approach**

The overall strategy for deriving a model architecture was to decrease validation error and make car drive smoothly.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it has proven effectiveness during sign classification.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by addding dropout layer.

Then I gather more data by making one more circle in both directions. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I made more training examples on that particular parts of the road. In my case, it was near the bridge.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

**2. Final Model Architecture**

The final model architecture (model.py lines 77-94) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x20x3 RGB image   							| 
| Lambda layer  		| Normalize image and mean centering   			| 
| Convolution 24x5x5   	| 2x2 stride, valid padding						|
| RELU					|												|
| Convolution 36x5x5   	| 2x2 stride, valid padding						|
| RELU					|												|
| Convolution 48x5x5   	| 2x2 stride, valid padding						|
| RELU					|												|
| Convolution 64x3x3   	| 1x1 stride, valid padding						|
| RELU					|												|
| Convolution 64x3x3   	| 1x1 stride, valid padding						|
| RELU					|												|
| Fully connected		| 1164 neurons									|
| RELU					|												|
| Dropouts				| 50%											|
| Fully connected		| 100 neurons									|
| RELU					|												|
| Dropouts				| 50%											|
| Fully connected		| 50 neurons									|
| RELU					|												|
| Dropouts				| 50%											|
| Fully connected		| 10 neurons									|
| RELU					|												|
| Dropouts				| 50%											|
| Fully connected		| 1 neurons (result)							|


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

**3. Creation of the Training Set & Training Process**

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from wrong decisions and not to stay on the center all the time. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image51]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would create more training sets. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I have used images from Left and Right cameras. Becouse of different view angels I had to -0.2 and +0.2 to steering respectively.

After the collection process, I had X number of data points. I then preprocessed this data by:
* Remove unused top part of the image
* Normalize and mean centered images


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by:

![alt text][image8]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
