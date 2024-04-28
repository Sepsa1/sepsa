from tqdm.auto import tqdm
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import torch


def train(ResNetModel, AbMILPmodel, epoch, trainloader, optimizer, fold, sw, last_epoch, entropy):
    ResNetModel.train()
    AbMILPmodel.train()
    print('Training')
    counter = 0
    train_loss = 0.
    train_proper = 0.
    len_trainloader = len(trainloader)
    all_preds = []
    all_labels = []
    for batch_idx, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        if len(image.shape) == 2:
            len_trainloader -= 1
            continue
        image = torch.squeeze(image, dim=0)
        image = image.to(next(ResNetModel.parameters()).device)
        labels = labels.to(next(ResNetModel.parameters()).device)
        label = labels[0]

        ResNetOutputs = ResNetModel(image)
        ResNetOutputs, label = Variable(ResNetOutputs), Variable(label)
        optimizer.zero_grad()

        pred = AbMILPmodel(ResNetOutputs)[1]
        loss = AbMILPmodel.calculate_objective(ResNetOutputs, label, entropy)
        train_loss += loss.item()
        proper = AbMILPmodel.proper_values(ResNetOutputs, label)
        train_proper += proper
        loss.backward()
        optimizer.step()
        all_preds.append(pred.cpu().detach().numpy().item())
        all_labels.append(label.cpu().detach().numpy().item())

    train_loss /= len_trainloader
    accuracy = train_proper / len_trainloader * 100

    sw.add_scalar('train_loss' + str(fold), train_loss, epoch)
    sw.add_scalar('train_accuracy' + str(fold), accuracy, epoch)
    print('Epoch: {}, Train loss: {:.4f}, Train accuracy: {:.4f}%'.format(epoch, train_loss, accuracy))

    if epoch + 1 == last_epoch:
        cm = confusion_matrix(all_labels, all_preds, normalize='true')
        ConfusionMatrixDisplay(cm).plot()
        plt.savefig('confusion_matrix_train_' + str(fold) + '.jpg')


def validate(ResNetModel, AbMILPmodel, epoch, validloader, optimizer, fold, sw, last_epoch, entropy):
    ResNetModel.eval()
    AbMILPmodel.eval()
    print('Validation')
    counter = 0
    valid_loss = 0.
    valid_proper = 0.
    len_validloader = len(validloader)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(validloader), total=len(validloader)):
            counter += 1
            image, labels = data
            if len(image.shape) == 2:
                len_validloader -= 1
                continue
            image = torch.squeeze(image, dim=0)
            image = image.to(next(ResNetModel.parameters()).device)
            labels = labels.to(next(ResNetModel.parameters()).device)
            label = labels[0]

            ResNetOutputs = ResNetModel(image)
            ResNetOutputs, label = Variable(ResNetOutputs), Variable(label)
            optimizer.zero_grad()

            pred = AbMILPmodel(ResNetOutputs)[1]
            loss = AbMILPmodel.calculate_objective(ResNetOutputs, label, entropy)
            valid_loss += loss.item()
            proper = AbMILPmodel.proper_values(ResNetOutputs, label)
            valid_proper += proper

            all_preds.append(pred.cpu().detach().numpy().item())

            all_labels.append(label.cpu().detach().numpy().item())

    # calculate loss and error for epoch
    valid_loss /= len_validloader
    accuracy = valid_proper / len_validloader * 100

    sw.add_scalar('valid_loss' + str(fold), valid_loss, epoch)
    sw.add_scalar('valid_accuracy' + str(fold), accuracy, epoch)
    print('Epoch: {}, Valid loss: {:.4f}, Valid accuracy: {:.4f}%'.format(epoch, valid_loss, accuracy))

    if epoch + 1 == last_epoch:
        cm = confusion_matrix(all_labels, all_preds, normalize='true')
        ConfusionMatrixDisplay(cm).plot()
        plt.savefig('confusion_matrix_valid_' + str(fold) + '.jpg')
