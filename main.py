from __future__ import print_function
from abmil_dataset import Sepsis_dataset
from noise import GaussianNoise
from torchvision import models, transforms
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils.class_weight import compute_class_weight
from train_valid import train, validate
from torch.utils.data import ConcatDataset, DataLoader

import AbMILPmodel
import torch.nn as nn
import torch.optim as optim
import torch
import os
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if __name__ == '__main__':
    sw = SummaryWriter()
    noise = GaussianNoise(0, 1)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        v2.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation((0, 360)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        noise,
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    Train_Fold_1 = Sepsis_dataset("target_folder/Fold_1", train_transform)
    Train_Fold_2 = Sepsis_dataset("target_folder/Fold_2", train_transform)
    Train_Fold_3 = Sepsis_dataset("target_folder/Fold_3", train_transform)
    Train_Fold_4 = Sepsis_dataset("target_folder/Fold_4", train_transform)
    Train_Fold_5 = Sepsis_dataset("target_folder/Fold_5", train_transform)

    Test_Fold_1 = Sepsis_dataset("target_folder/Fold_1", test_transform)
    Test_Fold_2 = Sepsis_dataset("target_folder/Fold_2", test_transform)
    Test_Fold_3 = Sepsis_dataset("target_folder/Fold_3", test_transform)
    Test_Fold_4 = Sepsis_dataset("target_folder/Fold_4", test_transform)
    Test_Fold_5 = Sepsis_dataset("target_folder/Fold_5", test_transform)

    training_data = [ConcatDataset([Train_Fold_2, Train_Fold_3, Train_Fold_4, Train_Fold_5]),
                     ConcatDataset([Train_Fold_1, Train_Fold_3, Train_Fold_4, Train_Fold_5]),
                     ConcatDataset([Train_Fold_1, Train_Fold_2, Train_Fold_4, Train_Fold_5]),
                     ConcatDataset([Train_Fold_1, Train_Fold_2, Train_Fold_3, Train_Fold_5]),
                     ConcatDataset([Train_Fold_1, Train_Fold_2, Train_Fold_3, Train_Fold_4])]

    valid_data = [Test_Fold_1, Test_Fold_2, Test_Fold_3, Test_Fold_4, Test_Fold_5]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    modelResNet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = modelResNet.fc.in_features
    num_classes = 3
    epochs = 300

    for i in range(0, 5):
        train_loader = DataLoader(training_data[i], batch_size=1, shuffle=True, pin_memory=True, num_workers=10,
                                  persistent_workers=True, drop_last=True)
        valid_loader = DataLoader(valid_data[i], batch_size=1, shuffle=False, pin_memory=True, num_workers=10,
                                  persistent_workers=True)
        modelAbMILP = AbMILPmodel.Attention(True).to(device)
        for param in modelAbMILP.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(modelAbMILP.parameters(), lr=10e-5, betas=(0.9, 0.999), weight_decay=10e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5
                                                    , gamma=0.8)
        modelResNet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modelResNet.fc = nn.Linear(num_features, num_classes)
        modelResNet.load_state_dict(torch.load('model_fold' + str(i) + '.pth'))
        modelResNet = torch.nn.Sequential(*(list(modelResNet.children())[:-1]))
        for param in modelResNet.parameters():
            param.requires_grad = False
        modelResNet = modelResNet.to(device)

        train_labels = [x[1] for x in training_data[i]]
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels
        )

        criterion = nn.CrossEntropyLoss(torch.from_numpy(class_weights).float().to(device))

        for epoch in range(epochs):
            train(modelResNet, modelAbMILP, epoch, train_loader, optimizer, i, sw, epochs, criterion)
            validate(modelResNet, modelAbMILP, epoch, valid_loader, optimizer, i, sw, epochs, criterion)
