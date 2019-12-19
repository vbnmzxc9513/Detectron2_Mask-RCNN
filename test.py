import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2

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

from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset", {}, "test.json", "./test_images")

metadata = MetadataCatalog.get("my_dataset")
dataset_dicts = DatasetCatalog.get("my_dataset")

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.OUTPUT_DIR = "./output_X101_imagenet_augment5"
cfg.DATASETS.TRAIN = ("my_dataset",)
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # 3 classes (data, fig, hazelnut)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#print(cfg.OUTPUT_DIR)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("my_dataset")
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
from itertools import groupby

#for i ,d in enumerate(dataset_dicts[5:8]):
for i ,d in enumerate(random.sample(dataset_dicts, 3)):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(str(i)+'.jpg', v.get_image()[:, :, ::-1])
'''
im = cv2.imread('342549.jpg')
outputs = predictor(im)

v = Visualizer(im[:, :, ::-1],
               metadata=metadata,
               scale=0.8,
               instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
               )
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#print(outputs["instances"].to('cpu')._fields['pred_masks'][0].numpy())
#print(len(outputs["instances"]))

cv2.imwrite('IC_Lab.jpg', v.get_image()[:, :, ::-1])
'''