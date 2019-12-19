# Detectron2_Mask-RCNN
NCTU_VRDL

##Reference
Detecron2 Github(Mask-RCNN): https://github.com/facebookresearch/detectron2
CLoDSA Github(Augmentation): https://github.com/joheras/CLoDSA

##Quick Start
1.Install the detectron2 environment from https://github.com/facebookresearch/detectron2.
2.Put train_X101_FPN_ImageNet_augment.py into detectron2 folder.
3.Set config:
  a.first time start: Comment out the cfg.MODEL.WEIGHTS and it will initialize an Imagenet and training.
  b.set register_coco_instances("my_dataset", {}, "train_images3_annotation.json", "./train_images3") this line accourding your dataset       path and your annotation.json path.
  c.I set the cfg.SOLVER.BASE_LR = 0.00025.
  d.In this homework case, if you are first time training, cfg.SOLVER.MAX_ITER might need to close to 30000~40000.
4.Command line "python train_X101_FPN_ImageNet_augment.py" and start training.

cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml") is the Hyperparameter which set by office.
