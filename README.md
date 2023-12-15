# Field Plant Disease Detection

A model ensemble solution for plant disease detection in field images, addressing challenges faced by existing models. Leveraging the Segment Anything Model, Image Processing techniques, and a Fully Convolutional Data Description, the proposed model achieves a 15% improvement in validation accuracies on field plant datasets like PlantDoc. It provides an accurate and practical tool for identifying and classifying multiple diseases in a single-field plant image.

# Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

Researchers have designed various CNN trained on public or private datasets for plant disease detection to help farmers remedy crop yield losses on their farms because of plant diseases. Plant village is a publicly available and the most widely used plant disease dataset because it contains 54,309 laboratory images across 14 crop species and 38 plant disease classes. Laboratory images were captured under controlled conditions in laboratories with a single leaf on each image and a uniform background. Models trained on such datasets have very low accuracies when running on field images captured directly in plantations with various interwoven leaves, complex backgrounds, and different lighting conditions. Furthermore, existing studies in this area have not focused on multi-disease identification using a single-field plant image. In this study, we propose a model ensemble solution for the accurate identification and classification of plant diseases on field images. The model uses Segment Anything Model to efficiently circumscribe all identifiable objects in the image, and Image Processing techniques are used to isolate the identified object from the original image. To differentiate identified background objects from actual leaf objects, we used a Fully Convolutional Data Description, an explainable deep one-class classification model for anomaly detection. Finally, the selected Region of interest is submitted to a Plantvillage-trained classification model for inference. Our model can detect multiple diseases appearing on different leaves of the same image and improves validation accuracies by 15\% on public field plant disease datasets such as PlantDoc, thus providing a reliable solution for farmers and practitioners.

![Model Workflow](https://github.com/emmanuelmoupojou2/Field_Plant_Disease_Detection/blob/main/moupo25.PNG)

## Features

The key features of the model are:

- Properly segment the original input field plant image to identify the different objects
- Identify objects that are actual plant leaves
- Determine the diseases affecting the identified leaves

## Installation

Tests were performed in the following environment: python=3.9.15, tensorflow=2.11.0, torch=1.9.1 and torchvision=0.10.1
The model uses [Segment Anything Model](https://github.com/facebookresearch/segment-anything) by Meta AI Research and [Fully Convolutional Data Description](https://github.com/liznerski/fcdd).
All the necessary libs and packages will be installed with the following commands:

```bash
# Create a new environment <env> with python=3.9.15 tensorflow=2.11.0 
conda create -c conda-forge -n <env> python=3.9.15 tensorflow=2.11.0

# Activate the newly created environment
conda activate <env>

# Clone the repository
git clone <repo>

# Move to cloned repository
cd <repo>

# Install Segment Anything Model
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install Fully Convolutional Data Description
git clone https://github.com/liznerski/fcdd.git
cd fcdd/python/
pip install .
cd ../../

#Install additional packages
pip install imutils pandas
```
## Usage

For better performance, the model should be run on a 8 Gb RAM
```bash
# Use GPU if available
[CUDA_VISIBLE_DEVICES=3] python predict_disease.py <plant_disease_image_path> <number_objects>

#The results will show as a dictionnary:
{'path_to_object1': 'disease1', 'path_to_object2': 'disease2', etc.}
```
## License

The project is open source and distributed under the [MIT License](https://chat.openai.com/c/LICENSE)
