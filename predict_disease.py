from __future__ import annotations

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import scipy
import imageio, os, skimage
import cv2  # type: ignore
from PIL import Image, ImageDraw, ImageFilter
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse, json, os
from typing import Any, Dict, List
from PIL import Image
import skimage
from skimage import data, filters, measure, morphology, io
import numpy as np
from scipy import ndimage
import cv2, os
import matplotlib.pyplot as plt
import imutils
import sys
from run_prediction_with_snapshot import predict
from classification import predict_disease
import shutil
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageFile
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

global inference_model
inference_model = None


def save_largest_green_region(masks, filepath, output, filename, destination_path, image_number, leaves_to_use=1):
    predictions = None
    leaf_leaf_folder = output + "/leaf" + "/leaf"  # Directory containing individual objects with plantvillage background

    if (os.path.exists(leaf_leaf_folder)):
        shutil.rmtree(leaf_leaf_folder)
    os.makedirs(leaf_leaf_folder, exist_ok=True)

    if (os.path.exists(destination_path)):
        shutil.rmtree(destination_path)
    os.makedirs(destination_path, exist_ok=True)

    masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    predicted_leaves = 0

    for index in range(len(masks)):

        bbox_x0, bbox_y0, bbox_w, bbox_h = masks[index]["bbox"]

        if (bbox_w > 0 and bbox_h > 0):  # We make sure to process consistent regions
            masked_file_name = "";
            filtered_path = ""
            try:
                mask_data = masks[index]

                mask = mask_data["segmentation"]
                mask = mask * 255
                # print("mask = ", mask)
                masked_file_name = os.path.join(output, str(image_number) + "-" + str(index) + "-mask.png")

                cv2.imwrite(masked_file_name, mask)

                masked_image = Image.open(masked_file_name)
                mask_data["masked_image"] = masked_image  # we keep track of the masked image

                im1 = Image.open(filepath)
                fname = str(image_number) + "-" + str(index) + "-region.png"
                filtered_path = os.path.join(output, fname)
                mask_data["leaf"] = fname
                im2 = Image.new('RGB', im1.size, color=(255, 255, 255))  # The saved object will have white background
                im = Image.composite(im1, im2, masked_image.convert('L'))
                im = im.convert('RGB')
                image = im.crop((bbox_x0, bbox_y0, bbox_x0 + bbox_w, bbox_y0 + bbox_h))
                image.save((leaf_leaf_folder + "/" + fname))

                predicted_leaves += 1
                # break;
            except:
                print("Error fromarray: ", (leaf_leaf_folder + "/" + fname))

            if (os.path.exists(masked_file_name)):
                os.remove(masked_file_name)

    if (predicted_leaves > 0):
        classification_path = output + "/classification/leaves"
        if (os.path.exists(classification_path)):
            shutil.rmtree(classification_path)
        os.makedirs(classification_path, exist_ok=True)
        pred_test = predict(
            output + "/leaf")  # pred_test is dictionnary with file paths and anomaly scores sorted ascending
        leaf_region, collected_leaves = [], 0

        pred_test = dict(sorted(pred_test.items(), key=lambda item: item[1]))

        for leaf_file, anomaly_score in pred_test.items():
            image = cv2.imread(leaf_file)
            h, w, c = image.shape
            if (collected_leaves < leaves_to_use):  # No additional constraint

                final_path = classification_path + '/' + str(image_number) + "-" + str(collected_leaves) + ".png"
                leaf_region += [final_path]
                #print("leaf_file, final_path = ", leaf_file, final_path)
                shutil.copyfile(leaf_file, final_path)
                collected_leaves += 1
        try:
            # Predict disease classes in destination_path
            predictions = dict()
            leaf_region.sort()  # Sort alphabetically
            if (len(leaf_region) > 0):
                predicts = predict_disease(output + "/classification")
                for filename, prediction in zip(leaf_region, predicts):
                    #print("Prediction: ", filename, "=", prediction)
                    predictions[filename] = prediction
            #print("predictions = ", predictions)
        except Exception as e:
            print("Error during classification: ", str(e))

    # if (os.path.exists(output+"/classification")): #Files needed to send response to request
    #     shutil.rmtree(output+"/classification")

    if os.path.exists(leaf_leaf_folder):
        shutil.rmtree(leaf_leaf_folder)

    if os.path.exists(destination_path):
        shutil.rmtree(destination_path)

    leaf_folder = output + "/leaf"
    if os.path.exists(leaf_folder):
        shutil.rmtree(leaf_folder)

    return predictions

global generator
generator = None

def segment_and_predict(filepath, leaves_to_use):
    global generator
    if (generator == None):
        print("Loading model in segment_and_predict...")
        sam = sam_model_registry["vit_h"](checkpoint="model_weights/sam_vit_h_4b8939.pth")
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Device = ", DEVICE)
        sam = sam.to(device=DEVICE)
        generator = SamAutomaticMaskGenerator(sam)

    current_working_directory = os.getcwd()
    output_directory = current_working_directory + '/segmented_objects'
    os.makedirs(output_directory, exist_ok=True)
    targets = [[filepath, os.path.basename(filepath), output_directory]]
    # #Resize image to avoid out of memory errors during processing
    im2 = Image.open(filepath)
    w, h = im2.size
    if (w > 500 or h > 500):  # Redimensionnement de l'image si nécessaire
        try:
            im2 = im2.resize((500, 500))
            im2 = im2.convert('RGB')
            im2.save(filepath)
        except:
            print("Error while resizing: ", filepath)

    image = cv2.imread(filepath)
    if image is None:
        print(f"Could not load '{filepath}' as an image, skipping...")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        masks = generator.generate(image)

        return save_largest_green_region(masks, filepath, output_directory, os.path.basename(filepath),
                                         os.path.join(output_directory, os.path.basename(filepath)),
                                         1,
                                         leaves_to_use)

        
if __name__ == "__main__":
    filepath, leaves_to_use = sys.argv[1], int(sys.argv[2])
    print(segment_and_predict(filepath, leaves_to_use))

