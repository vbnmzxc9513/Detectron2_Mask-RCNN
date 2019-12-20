# Detectron2_Mask-RCNN
NCTU_VRDL

## Reference
Detecron2 Github(Mask-RCNN): https://github.com/facebookresearch/detectron2

CLoDSA Github(Augmentation): https://github.com/joheras/CLoDSA

# Implementation
## Quick Start
1.##Install the detectron2 environment from https://github.com/facebookresearch/detectron2. ##

2.Put train_X101_FPN_ImageNet_augment.py and all python files into detectron2 folder.

3.Set config:

    a.first time start: Comment out the cfg.MODEL.WEIGHTS and it will initialize an Imagenet and training.
  
    b.set register_coco_instances("my_dataset", {}, "train_images3_annotation.json", "./train_images3") this line accourding your dataset path and your annotation.json path.
  
    c.I set the cfg.SOLVER.BASE_LR = 0.00025.
  
    d.In this homework case, if you are first time training, cfg.SOLVER.MAX_ITER might need to close to 30000~40000.
  
4.Command line "python train_X101_FPN_ImageNet_augment.py" and start training.

(There are more example you can see in the detectron2 github)

cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml") is the Hyperparameter which set by office.

## Simple Test
1.Find the "file.pth" in the output folder.

2.Set cfg.MODEL.WEIGHTS path to target ".pth" file.

3.cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml") Hyperparameters corresponding to the model you want to test.

4.Use which line been command out in the bottom. Read your own picture and predict.

5.Finally, command python test.py.

## DEMO Example:
![image](https://github.com/vbnmzxc9513/Detectron2_Mask-RCNN/blob/master/11.jpg)

## Output the json to submission
1.Find the "file.pth" in the output folder.

2.Set cfg.MODEL.WEIGHTS path to target ".pth" file.

3.cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml") Hyperparameters corresponding to the model you want to test.

5.Finally, command python submission_file.py.


# Experience
## Image Augmentation

See Image_augmentation.ipynb. 

This will generate a new dataset with annotation after augmentation. More detail you could see https://github.com/joheras/CLoDSA.



## Introduction

  The pretrained ImageNet model which I used was download from "detectron2://ImageNetPretrained/FAIR/X-101-32x8d.pkl".
In order to solve the problem of fitting, I consulted a lot of information. And implemented L2, picture enhancement, observation test curve, etc. Many of them refer to this KAGGLE article written by the ten^th place contestant(https://www.kaggle.com/c/data-science-bowl-2018/discussion/56238).


## Chose Model

  Usually, a small amount of data is not suitable for complex models, but in the spirit of experiment, I try to use 3 different models to train 100,000 Images for 100,000 iterations. These three models include Mask-RCNN Resnet-50, Mask-RCNN Resnet-101, and ResNext-101.
It was unexpectedly found that the most complicated model, ResNext-101, training results achieved the highest score of 0.4934 in the mAP of the submission. Resnet-50 and Resnet-101 get score separately 0.45 and 0.46. So for the subsequent experiments, I used RESNEXT-101 as the main training model.

## Image Augmentation
I tried a lot of combination of the dataset augmentation. Includes all used and only used parts. But the results are similar. 

1. Flips: Flips image including horizontal and vertical.

2. Invert: Invert the image color. 

3. Bilateral blurring: Bilateral blurring. 

4. Gaussian blurring: Gaussian blurring. 

5. Gaussian noise: Add Gaussian noise. 

6. Average blurring: Average blurring. 

7. Blurring: A common Blurring. 

8. Dropout: Cause 0.05% of image pixels randomly dropout. 

9. Elastic deformation: Applies elastic deformation as explained in the paper: P. Simard, D. Steinkraus, and J. C. Platt. Best practices for convolutional neural networks applied to visual document analysis. Proceedings of the 12th International Conference on Document Analysis and Recognition (ICDAR'03) vol. 2, pp. 958--964. IEEE Computer Society. 2003. 

10. Gamma correction: Applies gamma correction to the images. 

11. Salt and Pepper: Add Salt and Pepper to images. 

12. Translation: Shift images.

13. Crop: Crops pixels at the sides of the image. 

14. Equalize histogram: Applies histogram equalization to the image.

I have implemented these image augmentations by the library, CLoDSA. But I didn’t get the effect very much. 

I figured out two reasons. The first is that I found that the detectron2 platform developed by Facebook already adopted strategies of flipping and rotating during training. The second is that the other strategies I use may not be realistic and therefore have no effect.

## Regularization L2:

Set __C.SOLVE.R.WEIGHT_DECA Y 0.000 5 this config to implement.

By adjusting the error function to aggravate the penalty caused by the parameters of the high power term, to limit the infinite increase of the power of the high power term to fit the training model.

## Image processing the test set:
I even do image processing to the test set. The method including histogram equalization and invert the color of image. But didn’t get better score.

