# Age_and_gender_detection_using_deep_learning

## DEEP LEARNING: 
   Deep learning is a subset of AI 

## MACHINE LEARNING:
  We see some mathematical algorithms in machine learning , we train machine using data how to interpolate into future through the data.
  for example, we have features of dog and features of cat, if we have givven only the features to the algorithm , then it can say the input is a dog or a cat.
  We can do classification , regression, prediction etc,
  for example , if we take example of house prediction, if we have data of several houses(number of rooms, no. of bedrooms, size of the room) 
                suddenly, if we get a new set of features , we can use the previous data to train the model and our model can detect new price of the new house.
  
## IMPORTANCE OF DEEP LEARNING:
Deep learning is inspired from human brain. Brain contains neurons. These neurons process the information and sends it from one cell to another cell using nerve impulses. In deep learning , our target is that we approximate anytype of complex function. In order to succeed , we use series of layers. Each layer contains neurons, Each neuron corresponds a value. Each laer contains one or more neurons. We buid a multilayer neural network, which facilitates to do all the work, After building the model, we feed input data to it(which updates its weights according to the feedback). This updation approximately reaches the actual ideal function, SO this is our target to come close to the function while updating the weights. As much we run our huge data in these neural networks, that much our weights gets adjusted and makes our model more accurate. We update these weights in these way:- we will be given a function, which takes the data as the input layer and finally gives output layer as output. It becomes very easy to classify or regress any incoming data using neural layers.

## WHY IS DEEP LEARNING BECOMING POPULAR:-
Now-a-days we have excellent computers, Ram, Cpus and Gpus. Keeping in this mind, deep learning enthusiasts are training a lot models and giving numerous insights. We have a lot of inbuilt libraries which provides a lot kind of networks, they are Tensorflow, Pytorch, Caffe, Microsoft Cognitivr toolkit, D4JS , etc Now a days its become very easy to play with neural nwtworks.

## STEPS:
### DATASET:
In the data set we have choosen (UTKFace), it contains nearly 24000 images.
### IMAGE PREPROCESSING :
We have extracted the gender labels embedded into image file name(age, gender, ethnicity).
We have converted these images to greyscale from RGB.(because of less resouces)
### CNN
To make our model fitting less computationally, next up we have used cnn based deep learning approach for model building.
CNNs are really well suited to handle image data and it has given a very good accuracy.
### KAGGLE ENVIRONMENT :
We required a high-end machine with GPU . So kaggle is providing a inbuilt gpu and its ve similar to jupyter notebook which made us use it.
### training and testing:
There are 23,708 images in UTK-Face dataset.
using split function these are divided into 18970(80%) images for training data set and 4740(20%) images for validation data set.
### model:
Using 30 epochs on our defined cnn architechture, comprising of 
1) an input Conv2D layer(with 32 filters) paired with an MaxPooling2D layer(activation function relu has been used for improving perfomance)
2) 3 layers of Conv2D (with 64, 128 and 256 filters respectively) and corresponding MaxPooling layers.
3) As there are two outputs( age and gender) 2 Dense layers with 256 nodes each were defined.
4) Two output layers one for gender output and other for age output.
5) Compiled the model with cross entropy for binary classification.
