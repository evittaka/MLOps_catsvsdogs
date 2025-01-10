# Project description
---

## Goal
This project is undertaken by Group 50 as part of the 02476 Machine Learning Operations course at DTU. The primary objective is to apply the principles and techniques learned during the course to solve a machine learning problem. We aim to leverage one of the provided frameworks to build a scalable and well-structured solution. As an outcome, we could expect to be gained a profficiency in MLOps tools such as, Docker, Hydra, WanDB, Cloud services and many more.

## Framework
For this project, we have chosen the TIMM (PyTorch Image Models) framework, specifically designed for computer vision tasks. TIMM offers a wide selection of pre-trained models that can be fine-tuned to suit our needs. This framework is particularly advantageous due to its robust integration with PyTorch, streamlined APIs, and access to cutting-edge architectures. Using TIMM, we will construct and experiment with state-of-the-art deep learning models. As part of this process, we will set up the TIMM framework within our development environment, ensuring that our codebase is clean, modular, and well-managed through version control.

## Data
The dataset selected for this project is the Cats vs Dogs dataset from Kaggle. This dataset consists of 25,000 labeled images of cats and dogs, making it ideal for a binary image classification task. The dataset is well-suited for our goal of exploring and benchmarking convolutional neural networks (CNNs). The images are diverse in terms of size, lighting, and orientation, which presents unique challenges for model training. Before feeding the data into our models, we will perform preprocessing steps such as resizing, normalization, and augmentation to enhance model performance. 

## Models
Our task is to perform binary image classification on the Cats vs Dogs dataset. Initially, we will develop a baseline CNN model to establish a point of reference. Subsequently, we will explore and compare the performance of advanced models available in the TIMM framework, including ConvNeXt, MobileNet V3, ResNet, and VGG. This comparison will help us understand the strengths and trade-offs of each model in terms of accuracy, computational efficiency, and scalability.
