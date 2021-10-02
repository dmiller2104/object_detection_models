# example of extracting bounding boxes from an annotation file
from xml.etree import ElementTree
import numpy as np
from mrcnn.utils import Dataset, extract_bboxes, compute_ap
from mrcnn.visualize import display_instances
from mrcnn.model import MaskRCNN
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.config import Config
from os import listdir
import matplotlib.pyplot as plt
from numpy.lib.shape_base import expand_dims

#%%
# class that defines and loads the kangaroo dataset
class KangarooDataset(Dataset):
	# load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
		# define one class
	    self.add_class("dataset", 1, "kangaroo")
		# define data locations
	    images_dir = dataset_dir + '/images/'
	    annotations_dir = dataset_dir + '/annots/'
		# find all images
	    for filename in listdir(images_dir):
			# extract image id
		    image_id = filename[:-4]
			# skip bad images
		    if image_id in ['00090']:
			    continue
			# skip all images after 150 if we are building the train set
		    if is_train and int(image_id) >= 150:
			    continue
			# skip all images before 150 if we are building the test/val set
		    if not is_train and int(image_id) < 150:
			    continue
		    img_path = images_dir + filename
		    ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
		    self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
 

    # function to extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load xml
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = np.zeros([h, w, len(boxes)], dtype = 'uint8')
        # create masks
        class_ids = []
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('kangaroo'))
        return masks, np.asarray(class_ids, dtype = 'int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

#%%
# define a configuration for the model
class KangarooConfig(Config):
	# define the name of the configuration
	NAME = "kangaroo_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 1
	# number of training steps per epoch
	STEPS_PER_EPOCH = 131

# %%
# evaluating the model using Mean Average Precision (MAP)
# we are predicting bounding boxes
# so to get accuracy we compare the one drawn by the model versus the actual boxes
# a perfect score will be 1
# pos score > 0.5, also using precision/avg precision (AP) and recall
class PredictionConfig(Config):
    # define the name of the config
    NAME = 'kangaroo_cfg'
    # n classes
    NUM_CLASSES = 1 + 1
    # simplify the GPU count
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

#%%
def evaluate_model(dataset, model, cfg):
    APs = []
    for image_id in dataset.image_ids:
        #load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values
        scaled_image = mold_image(image, cfg)
        # convet image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose = 0)
        # extract results
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = np.mean(APs)
    return mAP

#%%
# train set
train_set = KangarooDataset()
train_set.load_dataset('kangaroo', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
 
# test/val set
test_set = KangarooDataset()
test_set.load_dataset('kangaroo', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

#%%
image_id = 10
image = train_set.load_image(image_id)
print(image.shape)
# load image mask
mask, class_ids = train_set.load_mask(image_id)
print(mask.shape)

#%%
# plot image
plt.imshow(image)
# plot mask
plt.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
plt.show()

# %%
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    image = train_set.load_image(i)
    plt.imshow(image)
    # plot all masks
    mask, _ = train_set.load_mask(i)
    for j in range(mask.shape[2]):
        plt.imshow(mask[:, :, j], cmap = 'gray', alpha= 0.3)
#show the figure
plt.show()
# %%
# printing all of the image info objects to the console
# helps to confirm that all of the calls to the add_image() func in the load_dataset() func worked
for image_id in train_set.image_ids:
    # load image info
    info = train_set.image_info[image_id]
    # display on the console
    print(info)
# %%

# define image id
image_id = 99
# load the image
image = train_set.load_image(image_id)
# load the masks and the class ids
mask, class_ids = train_set.load_mask(image_id)
# extract bounding boxes from the masks
bbox = extract_bboxes(mask)
# display image with masks and bounding boxes
display_instances(image, bbox, mask, class_ids, train_set.class_names)

#%%
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode = 'inference', model_dir = './', config=cfg)

#%%
# load the weights from the model we have produced 
model.load_weights('mask_rcnn_kangaroo_cfg_0001.h5', by_name=True)

#%%
train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)

#%%
# evalute the model
test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)

