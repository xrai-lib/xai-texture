# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import torch

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
#from mmseg.registry import RUNNERS
from mmseg.registry import DATASETS
from mmseg.apis import init_model, inference_model
import cv2

from mmseg.datasets import BaseSegDataset
import config
from utils import calculate_iou, add_to_test_results_unet, calculate_iou_unet, calculate_pixelaccuracy
from preprocessing.preprocess_unet import preprocess_dataset_unet_from_patches

import os
from PIL import Image

def train_unet(dataset, train_images_dir, test_images_dir, mask_images_dir):
    preprocess_dataset_unet_from_patches()
    classes=('background', 'cancer')
    palette=[[0, 0, 0], [128, 128, 128]]
    
    @DATASETS.register_module()
    class StanfordBackgroundDataset(BaseSegDataset):
      METAINFO = dict(classes = classes, palette = palette)
      def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
    
    cfg = config.unet_config_file_path
    
    cfg = Config.fromfile(cfg)
    launcher = 'pytorch'
    
    cfg.train_dataloader = dict(
        batch_size=16,
        dataset=dict(
            ann_file=os.path.join(config.data_path,'train.txt'),
            data_prefix=dict(
                # Change the img_path and seg_map_path for different feature dataset accordingly
                img_path=train_images_dir, seg_map_path=mask_images_dir),
            data_root=config.data_path,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    keep_ratio=True,
                    ratio_range=(
                        0.5,
                        2.0,
                    ),
                    scale=(
                        256,
                        256,
                    ),
                    type='RandomResize'),
                dict(
                    cat_max_ratio=0.75, crop_size=(
                        256,
                        256,
                    ), type='RandomCrop'),
                dict(prob=0.5, type='RandomFlip'),
                dict(type='PhotoMetricDistortion'),
                dict(type='PackSegInputs'),
            ],
            type='StanfordBackgroundDataset'),
        num_workers=4,
        persistent_workers=True,
        sampler=dict(shuffle=True, type='InfiniteSampler'))
    
    cfg.val_dataloader = dict(
        batch_size=1,
        dataset=dict(
            ann_file=os.path.join(config.data_path,'val.txt'),
            data_prefix=dict(
              # Change the img_path and seg_map_path for different feature dataset accordingly
                img_path=test_images_dir, seg_map_path=mask_images_dir),
            data_root=config.data_path,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(keep_ratio=False, scale=(
                    256,
                    256,
                ), type='Resize'),
                dict(type='LoadAnnotations'),
                dict(type='PackSegInputs'),
            ],
            type='StanfordBackgroundDataset'),
        num_workers=4,
        persistent_workers=True,
        sampler=dict(shuffle=False, type='DefaultSampler'))
    cfg.train_cfg = dict(max_iters=16000, type='IterBasedTrainLoop', val_interval=1000)
    
    cfg.test_dataloader = cfg.val_dataloader
    
    cfg.default_hooks = dict(
        checkpoint=dict(by_epoch=False, interval=1000, type='CheckpointHook'))
    cfg.dataset_type = 'StanfordBackgroundDataset'
    cfg.work_dir = dataset
    
    #Train
    runner = Runner.from_cfg(cfg)
    runner.train()


def test_unet(result_path, choice, test_images_dir, mask_images_dir):
   # List all files in the input directory
   image_files = os.listdir(test_images_dir)
   config_file = config.saved_models_path + '/UNET/Feature_' + str(choice)+'/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py'
   checkpoint_file = config.saved_models_path + '/UNET/Feature_' + str(choice)+'/iter_16000.pth'
   device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
   model = init_model(config_file, checkpoint_file, device=device)
   total_iou = 0.0
   total_images = 0
   total_accuracy = 0.0
   # Process each image
   for image_file in image_files:
       # Skip non-image files
       if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
           continue
       
       # Construct the path to the input image
       input_image_path = os.path.join(test_images_dir, image_file)
       
       # Forward pass the input image through the model
       result = inference_model(model, input_image_path)
       
       # Convert predicted segmentation mask tensor to numpy array
       numpy_image = result.pred_sem_seg.data[0].cpu().numpy()
       
       # Load ground truth segmentation mask
       gt_mask_path = os.path.join(mask_images_dir, os.path.splitext(image_file)[0] + '.png')
       gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

       gt_mask[gt_mask == 255] = 1
       # Calculate IoU for the current image
       iou = calculate_iou_unet(numpy_image, gt_mask)

       acc = calculate_pixelaccuracy(numpy_image,gt_mask)
       
       # Update total IoU and total number of images
       total_iou += iou
       total_accuracy += acc
       total_images += 1


   # Calculate mean IoU
   if total_images > 0:
       average_iou = total_iou / total_images
       print("Mean IoU:", average_iou)
       mean_accuracy = total_accuracy / total_images
       print("Mean Accuracy:", mean_accuracy)
       add_to_test_results_unet(result_path, 'Feature_'+str(choice), average_iou, mean_accuracy)

   else:
       print("No images found for evaluation.")
       
   print(f"Testing Successful. Results added to " + result_path)
