# Object Detection Models

## A repository of object detection models - from basic onwards

The kangaroo object detection model in this repository is based on the work found at this link:
https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/

Object detection model uses Region-Based Convolutional Neural Networks (R-CNN). It uses object segmentation to localise objects.

## Command line codes

For the kangaroo object detection model you would need to run the following in the command line.

git clone https://github.com/matterport/Mask_RCNN.git
git clone https://github.com/experiencor/kangaroo.git

However, as both are included in this repository you can skip them. However you will need to execute the following:

cd Mask_RCNN
python setup.py install

The dataset is in kangaroo folder and contains images and annotations (which gives pixel locations for the roos).

The mask-rcnn library requres that train, validation and test datasets be managed by a mrcnn object. 

Please download the model weights for the pre-fit Mask R-CNN model from here:
https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

As part of this repository I have uploaded two h5 files which can be used as weights for the model. These files are:
mask_rcnn_kangaroo_cfg_0001.h5 
mask_rcnn_kangaroo_cfg_0002.h5
