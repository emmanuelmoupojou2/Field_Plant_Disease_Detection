# Field Plant Disease Detection

A model ensemble solution for plant disease detection in field images, addressing challenges faced by existing models. Leveraging the [Segment Anything Model](https://github.com/facebookresearch/segment-anything), Image Processing techniques, and [Fully Convolutional Data Description](https://github.com/liznerski/fcdd). The proposed model achieves a 15% improvement in validation accuracies on field plant datasets like PlantDoc. It provides an accurate and practical tool for identifying and classifying multiple diseases in a single-field plant image.

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
- Determine the diseases affecting the identified leaves. The predicted disease is one of the 38 disease classes of the [Plantvillage](https://github.com/gabrieldgf4/PlantVillage-Dataset) dataset: ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                   'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry___healthy',
                   'Corn___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
                   'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                   'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                   'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                   'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite',
                   'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

## Installation

Tests were performed in the following environment: python=3.9.15, tensorflow=2.11.0, torch=1.9.1 and torchvision=0.10.1

The model uses [Segment Anything Model](https://github.com/facebookresearch/segment-anything) by Meta AI Research and [Fully Convolutional Data Description](https://github.com/liznerski/fcdd).

All the necessary libs and packages will be installed with the following commands:


### Create a new environment <env> with python=3.9.15 tensorflow=2.11.0 
```bash
conda create -c conda-forge -n env python=3.9.15 tensorflow=2.11.0

conda activate env
```

### Clone the repository and Install [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
```bash
git clone <repo>
cd <repo>

pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Install [Fully Convolutional Data Description](https://github.com/liznerski/fcdd)
```bash
git clone https://github.com/liznerski/fcdd.git
cd fcdd/python/
pip install .
cd ../../
```

### Install additional packages
```bash
pip install imutils pandas
```

### Complete installation with SAM's checkpoint

Download the [Segment Anything Model's checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)s in the folder model_weights.
```bash
curl -l "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth">model_weights/sam_vit_h_4b8939.pth
```
## Usage

For better performance, the model should be run on 16 GB RAM
```bash
# Use GPU if available
[CUDA_VISIBLE_DEVICES=3] python predict_disease.py <plant_disease_image_path> <number_objects>

#The results will show as a dictionary:
{'path_to_object1': 'disease1', 'path_to_object2': 'disease2', etc.}
```
## License

The project is open source and distributed under the [MIT License](https://github.com/emmanuelmoupojou2/Field_Plant_Disease_Detection/tree/main?tab=MIT-1-ov-file)
