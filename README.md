# Object-Detection
Detection of objects using sliding window and pyramid image processing

## Procedure:
#### pyramid image processing:
Image is created by reducing the size with given scale and the cration of scaled images continues untill it reaches Roi size.

#### Sliding Window process:
Sliding image is a process where image with roi size is slided every step with sliding pixels (Which are referred to strides in CNN)

#### Object Detection:
All the images collected (slided images of all the pyramid images) and sent to object detection model (used ResNet50) and get he prediction parameters. We will create bounding boxes with prediction co-ordinates.

#### Non-Maxima Suppression:
This process Deletes duplicate Bounding boxes on the image by calculating the ration of overlap among the boxes of same label.
