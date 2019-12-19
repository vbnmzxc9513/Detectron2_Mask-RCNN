import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
import glob

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
import random
import cv2
import os

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.OUTPUT_DIR = "./output_X101_imagenet_augment"
cfg.DATASETS.TRAIN = ("my_dataset",)
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # 3 classes (data, fig, hazelnut)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_0189999.pth')
#print(cfg.OUTPUT_DIR)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("my_dataset")
predictor = DefaultPredictor(cfg)

cocoGt = COCO("test.json")

from utils import binary_mask_to_rle

coco_dt = []

for imgid in cocoGt.imgs:
    image = cv2.imread("test_images/" + cocoGt.loadImgs(ids=imgid)[0]['file_name'])[:,:,::-1] # load image
    outputs = predictor(image)
    n_instances = len(outputs["instances"])
    #print(n_instances)
    if n_instances > 0: # If any objects are detected in this image
        for i in range(n_instances): # Loop all instances
            # save information of the instance in a dictionary then append on coco_dt list
            pred = {}
            pred['image_id'] = imgid # this imgid must be same as the key of test.json
            pred['category_id'] = int(outputs["instances"]._fields['pred_classes'][i]) + 1
            #pred['segmentation'] = binary_mask_to_rle(masks[:,:,i]) # save binary mask to RLE, e.g. 512x512 -> rle
            pred['segmentation'] = binary_mask_to_rle(outputs["instances"].to('cpu')._fields['pred_masks'][i].numpy())
            pred['score'] = float(outputs["instances"]._fields['scores'][i])
            coco_dt.append(pred)

with open("submission.json", "w") as f:
    json.dump(coco_dt, f)