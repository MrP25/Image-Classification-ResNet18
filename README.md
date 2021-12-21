# Image-Classification-ResNet18
  A simple image classification task, using resnet18 model
# Introduction
  This is a simple image classification task, can easily used in your own datasets. The precession in my datasets is about 95% when lr=0.01 and batchsize=32. Recommand to use models to pretrain. 
# Using
  First please change the path of datasets and model save path or add dataset to directory. The default storage path is as follows:
    data_dir = "./datasets"
    model_save = "./resnet.pth"
 # requirements
  python=3.7
  torch=1.4
  torchvision=0.8
  numpy=1.17
