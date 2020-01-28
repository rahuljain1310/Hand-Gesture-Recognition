# Hand-Gesture-Recognition
The task in this assignment is to identify the direction of the gesture made by a user in front of the camera. 

## Dataset Generation
Images for each gesture were taken in multiple in multiple pose, lighting, background, scale etc. Dataprepare.py script takes an video input and segrates it the video into datasets folder.

We created 3 datasets for each gesture : Test, Training and Cross-Validation. Training Class consisting of nearly 3,100 images for each class whereas cross-validation and test dataset consists of 400 and 200 images respectively. 

Script consists of functions to resize, crop, add text , add rectangle etc. Various modifications are done on the data such as background subtraction, edge
detection, grayscale etc.

Dataset Link : https://drive.google.com/file/d/1yEdbo3AuS2ug3S9-daNMGr_FZ-HZFHrv/view?usp=sharing

## Accuracy, Training, Cross-Validation vs Epoch

The blue line denotes the training accuracy while the orange line denotes the cross-validation accuracy. We can see that cross -validation accuracy and training accuracy are both high after epoch 7. After this point, data is being overfitted into the model. Model state after epoch 7 is supposed to give best accuracy on test dataset.

<!-- <img src="https://rahuljain1310.github.io/RahulJainIITD/images_online/HandGesture_Accuracy.jpg"> -->

## Optimization / HyperParameter Tuning

1. Batch-Gradient Descent was used with mini-batch of size 32 as it gave better results compared to batches of size 16,64,24,48.
2. Total of 16 epochs were run:
<br> a. For the first 8 epoch, learning rate is 0.001 and momentum 0.9 
<br> b. In the next 4 epoch, learning rate is 0.0005 and momentum 0.6
<br> c. In the next 4 epoch, learning rate and momentum is further reduced to fine tune the CNN Model.
3. Cross Entropy Loss is used as the criteria for optimizing the model.
4. Threshold for the background class was tuned by testing.

<img src="https://rahuljain1310.github.io/RahulJainIITD/images_online/HandGesture_Architecture.jpg">

## Pre-processing Frame

1. Algorithm to get the mask:
<br>  a. Background subtraction: Either KNN or MOG2
<br>  b. Skin Detection using a pixel based decision tree: Combination of skin detection and background subtraction 

2. The mask is used to get contours and identify bounding box
<br> a. For each box the inference is made, if it exceeds a threshold then either left, right or stop is predicted, otherwise STOP is predicted.