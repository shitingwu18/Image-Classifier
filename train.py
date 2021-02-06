# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu

# Imports here
import os
import torchvision
import torch
import torchvision.models as models
import argparse
import time
from torch import optim
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import OrderedDict


def init_argparse():
    parser = argparse.ArgumentParser(description = 'add argument to train model')
    # Argument : Data directory
    parser.add_argument('data_dir', action = 'store', type = str, ### default = './flowers',
                    help = 'data directory of the model')
    ### # Argument : Choose architecture
    parser.add_argument('--arch', action = 'store', dest = 'arch', type = str, default = 'vgg16_bn')
    # Argument : Set directory to save checkpoints

    parser.add_argument('--save_dir', action = 'store',dest = 'save_dir', type = str, default = './checkpoint.pth',
                        help = 'path to save checkpoint')
    # Argument : Set hyperparameters
    parser.add_argument('--learning_rate', action = 'store',dest = 'learning_rate', type = float, default = 0.003)
    parser.add_argument('--hidden_units', action = 'append',dest = 'hidden_units', type = int, default = 1024)
    parser.add_argument('--epochs', action = 'store', dest = 'epochs', type = int, default = 45)
    # Argument : User GPU for training
    parser.add_argument('--gpu',action = 'store_true', dest = 'gpu', default = True)
    parser.add_argument('--barch_size', action = 'store', dest = 'batch_size', default = 64)
    args = parser.parse_args()

    return args
    ### Assign variable in_args to parse_args()

def train_model(data_dir, save_dir, learning_rate, hidden_units, epochs, gpu, batch_size, arch):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([transforms.RandomRotation(30)
                                        ,transforms.RandomResizedCrop(224)
                                        ,transforms.RandomHorizontalFlip()
                                        ,transforms.ToTensor()
                                        ,transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                                          ])


    test_transform = transforms.Compose([transforms.Resize(255)
                                        ,transforms.CenterCrop(224)
                                        ,transforms.ToTensor()
                                        ,transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                                          ])


    # TODO: Load the datasets with ImageFolder
    train_datasets = torchvision.datasets.ImageFolder(root = train_dir, transform = train_transform)
    valid_datasets = torchvision.datasets.ImageFolder(root = valid_dir, transform = test_transform)
    test_datasets = torchvision.datasets.ImageFolder(root = test_dir, transform = test_transform)


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloaders = DataLoader(train_datasets,batch_size=batch_size,shuffle=True)
    validloaders = DataLoader(valid_datasets, batch_size = batch_size, shuffle = True)
    testloaders = DataLoader(test_datasets, batch_size = batch_size, shuffle = True)


    ### Define model
    # TODO: Build and train your network
    # vgg = models.vgg16_bn(pretrained=True)
    vgg = getattr(models, arch)(pretrained = True)
    ### froze parameter
    for param in vgg.parameters():
        param.requires_grad = False
    ### define classifier for model
    classifier = nn.Sequential(OrderedDict(
    [('fc1', nn.Linear(25088, 4096))
    ,('relu1', nn.ReLU())
    ,('dropout', nn.Dropout(0.2))
    ,('fc2', nn.Linear(4096, 2056))
    ,('relu2', nn.ReLU())
    ,('fc3', nn.Linear(2056, hidden_units))
    ,('relu3', nn.ReLU())
    ,('fc4', nn.Linear(hidden_units, 102))
    ,('output', nn.LogSoftmax(dim = 1))
    ]
    ))
    vgg.classifier = classifier

    optimizer = optim.Adam(vgg.classifier.parameters(), lr = learning_rate)
    criterion = nn.NLLLoss()

    device = torch.device("cuda" if (torch.cuda.is_available() == True) & (gpu == True) else "cpu")
    print(device)
    ### Train model
    vgg.to(device)
    every_batch =5
    iteration = 0

    print('start training')
    for i in range(epochs):
        running_loss = 0
        accuracy = 0
        for images, labels in trainloaders:

            images, labels = images.to(device), labels.to(device)

            iteration += 1
            print('data to device')
            optimizer.zero_grad()

            log_ps = vgg.forward(images)
            print('finish feed forwrad')
            loss = criterion(log_ps, labels)

            loss.backward()
            optimizer.step()
            print('finish back propagation')
            running_loss += loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        vgg.eval()
        with torch.no_grad():
            print('start validation')
            valid_running_loss = 0
            valid_accuracy = 0
            for valid_images,valid_labels in validloaders:
                valid_images, valid_labels = valid_images.to(device), valid_labels.to(device)
                valid_log_ps = vgg.forward(valid_images)
                valid_loss = criterion(valid_log_ps, valid_labels)
                valid_running_loss += valid_loss.item()

                valid_ps = torch.exp(valid_log_ps)
                valid_top_p, valid_top_class = valid_ps.topk(1, dim = 1)
                #         print(top_class)
                valid_equals = valid_top_class == valid_labels.view(*valid_top_class.shape)
                valid_accuracy += torch.mean(valid_equals.type(torch.FloatTensor)).item()
        vgg.train()

        print('epoch: ', i + 1)
        print('train loss of this epoch: ', running_loss/len(trainloaders))
        print('accuracy of the epoch: ', accuracy/len(trainloaders))
        print('valid loss of this epoch: ', valid_running_loss/len(validloaders))
        print('accuracy of the valid set: ', valid_accuracy/len(validloaders))


    # # TODO: Save the checkpoint
    vgg.class_to_idx = train_datasets.class_to_idx
    # # vgg

    checkpoint = {

                    'classifier' : vgg.classifier
                    ,'epochs' : epochs
                    ,'class_to_idx' : vgg.class_to_idx
                    ,'state_dict' : vgg.state_dict()
                    ,'optimizer_state_dict' : optimizer.state_dict()
                    }
    torch.save(checkpoint, save_dir)

    return checkpoint

def main():
    start_time = time.time()
    ### input argument
    in_args = init_argparse()
    print(in_args)

    train_model(data_dir, save_dir, learning_rate, hidden_units, epochs, gpu, batch_size, arch)
        # TODO 0: Measure total program runtime by collecting end time
    end_time = time.time()

    # TODO 0: Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time #calculate difference between end time and start time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )


# Call to main function to run the program
if __name__ == "__main__":
    main()
