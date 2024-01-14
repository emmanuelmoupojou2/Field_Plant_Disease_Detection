import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from fcdd.training.fcdd import FCDDTrainer
from fcdd.models.fcdd_cnn_224 import FCDD_CNN224_VGG_F
from fcdd.datasets.image_folder import ImageFolder
from fcdd.datasets.preprocessing import local_contrast_normalization
from fcdd.util.logging import Logger
import torch

global inference_model
inference_model=None

def predict(images_path):

    global inference_model

    snapshot = "model_weights/snapshot.pt"

    net = FCDD_CNN224_VGG_F((3, 224, 224), bias=True).cuda() if torch.cuda.is_available() else FCDD_CNN224_VGG_F((3, 224, 224), bias=True).cpu()

    normal_class = 0

    # Use the same test transform as was used for training the snapshot (e.g., for mvtec, per default, the following)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize( [0.4426, 0.4850, 0.3583],[0.2114, 0.1987, 0.2214] )
    ])
    logger = None
    quantile = 0.97

    # Create a trainer to use its loss function for computing anomaly scores
    ds = ImageFolder(images_path, transform, transforms.Lambda(lambda x: 0))

    #loader = DataLoader(ds, batch_size=16, num_workers=0)#Can't list all file names with this option
    loader = DataLoader(ds, batch_size=1, num_workers=1)

    if(inference_model is None):
        inference_model = FCDDTrainer(net, None, None, (None, None), logger, 'fcdd', 8, quantile, 224)
        inference_model.load(snapshot)
        inference_model.net.eval()

    all_anomaly_scores, all_inputs, all_labels, i, all_file_names = [], [], [], 0, []
    for inputs, labels in loader:
        sample_fname, _ = loader.dataset.samples[i]
        all_file_names.append(sample_fname)
        i = i + 1
        inputs = inputs.cuda() if torch.cuda.is_available() else  inputs.cpu()
        with torch.no_grad():
            outputs = inference_model.net(inputs)
            anomaly_scores = inference_model.anomaly_score(inference_model.loss(outputs, inputs, labels, reduce='none'))
            anomaly_scores = inference_model.net.receptive_upsample(anomaly_scores, reception=True, std=8, cpu=False)
            all_anomaly_scores.append(anomaly_scores.cpu())
            all_inputs.append(inputs.cpu())
            all_labels.append(labels)

    all_inputs = torch.cat(all_inputs)
    all_labels = torch.cat(all_labels)

    all_anomaly_scores = torch.cat(all_anomaly_scores)

    # transform the pixel-wise anomaly scores to sample-wise anomaly scores
    final_anomaly_scores = inference_model.reduce_ascore(all_anomaly_scores)
    result = dict()
    for file, score in zip(all_file_names, final_anomaly_scores.tolist()):
        result[file] = score
    sorted_result = dict(sorted(result.items(), key=lambda item: item[1]))

    
    return sorted_result

