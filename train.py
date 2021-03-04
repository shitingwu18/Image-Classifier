# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu

# Imports here
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
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
import matplotlib.pyplot as plt


def init_argparse():
    parser = argparse.ArgumentParser(description = 'add argument to train model')
    # Argument : Data directory
    parser.add_argument('--data_dir', action = 'store', dest = 'data_dir', type = str, ### default = './flowers',
                    help = 'data directory of the model')
    ### # Argument : Choose architecture
    parser.add_argument('--arch', action = 'store', dest = 'arch', type = str, default = 'vgg19')
    # Argument : Set directory to save checkpoints

    parser.add_argument('--save_dir', action = 'store',dest = 'save_dir', type = str,
                        help = 'path to save checkpoint')
    # Argument : Set hyperparameters
    parser.add_argument('--learning_rate', action = 'store',dest = 'learning_rate', type = float, default = 0.003)
    parser.add_argument('--hidden_units', action = 'append',dest = 'hidden_units', type = int, default = 1024)
    parser.add_argument('--epochs', action = 'store', dest = 'epochs', type = int, default = 45)
    # Argument : User GPU for training
    parser.add_argument('--gpu',action = 'store_true', dest = 'gpu', default = True)
#     parser.add_argument('--loss', action = 'store', dest = 'loss', default = 'MSE')
    parser.add_argument('--barch_size', action = 'store', dest = 'batch_size', default = 64)
    args = parser.parse_args()

    ### adding help message
    if(args.arch == 'help'):
        print('List of available CNN networks:')
#         print('1. vgg11')
#         print('2. vgg13')
        print('1. vgg19')
#         print('4. vgg16_bn')
        print('2. desenet121')
        print('3. alexnet')
        quit()

    if(args.learning_rate > 1 or args.learning_rate < 0):
        print('Error: Invalid learning rate')
        print('Must be between 0 and 1 exclusive')
        quit()

    if(args.batch_size < 0):
        print('Error: Invalid batch size')
        print('Must be greater than 0')
        quit()

    if(args.hidden_units <= 0):
        print('Error: Invalid number of hidden units given, Must be greater than 0')
        quit()

    arches = ['vgg19', 'alexnet', 'densenet'] ###'vgg11', 'vgg13', 'vgg16', ]
#     lossF = ['L1', 'NLL', 'Poisson', 'MSE', 'Cross']

    if args.arch not in arches:
        print('Error: Invalid architecture naume received')
        print('Type \'python train.py -a help\' for more information')
        quit()

#     if args.loss not in lossF:
#         print('Error: Invalid architecture name received')
#         print('type \'python train.py -l help\' for more information ')
#         quit()

#     if args.device not in ['cpu', 'gpu']:
#         print('Error: invalid device name received')
#         print('It must be either \'cpu\' or \'gpu\'')
#         quit()

    return args
    ### Assign variable in_args to parse_args()

def loaders(data_dir, batch_size, transform):

    datasets = torchvision.datasets.ImageFolder(root = data_dir, transform = transform)
    dataloaders = DataLoader(datasets, batch_size = batch_size, shuffle = True)

    return datasets, dataloaders

def create_checkpoint(model, path, model_name, class_to_idx, optimizer, epochs):
    model.cpu()
    model.class_to_idx = class_to_idx
    checkpoint = {
                    'arch' : model_name,
                    'state_dict' : model.state_dict(),
                    'class_to_idx' : model.class_to_idx,
                    'epochs' : epochs,
                    'optimizer_state_dict' : optimizer.state_dict
    }
    if model_name == 'resnet101':
        checkpoint['fc'] = model.fc
    else:
        checkpoint['classifier'] = model.classifier
    file = str()
    if path != None:
        file = path + '/' + model_name + '_checkpoint.pth'
    else:
        file = model_name + '_checkpoint.pth'
    torch.save(checkpoint, file)
    print('Model(%s) has been saved into Path(%s)' % (model_name,file))

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


    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_datasets, trainloaders = loaders(train_dir,batch_size=batch_size,transform = train_transform)
    _ , validloaders = loaders(valid_dir,batch_size=batch_size,transform = test_transform)
    _ , testloaders = loaders(test_dir,batch_size=batch_size,transform=test_transform)


    # ### Define model
    # # TODO: Build and train your network
    # # vgg = models.vgg16_bn(pretrained=True)
    # vgg = getattr(models, arch)(pretrained = True)
    # ### froze parameter
    # for param in vgg.parameters():
    #     param.requires_grad = False
    # ### define classifier for model
    # classifier = nn.Sequential(OrderedDict(
    # [('fc1', nn.Linear(25088, 4096))
    # ,('relu1', nn.ReLU())
    # ,('dropout', nn.Dropout(0.2))
    # ,('fc2', nn.Linear(4096, 2056))
    # ,('relu2', nn.ReLU())
    # ,('fc3', nn.Linear(2056, hidden_units))
    # ,('relu3', nn.ReLU())
    # ,('fc4', nn.Linear(hidden_units, 102))
    # ,('output', nn.LogSoftmax(dim = 1))
    # ]
    # ))
    # vgg.classifier = classifier
    # optimizer = optim.Adam(vgg.classifier.parameters(), lr = learning_rate)

    t_models = {
                'vgg19': models.vgg19(pretrained = True),
                'densenet121': models.densenet121(pretrained = True),
                'resnet101': models.resnet101(pretrained = True)
    }

    model = t_models.get(arch, models.vgg19(pretrained = True))
    print(model)
    classifier = None
    optimizer = None

    if arch == 'vgg19':
        classifier = nn.Sequential(nn.Linear(25088, 4096),
                                nn.ReLU(),
                                nn.Dropout(0.4),
                                nn.Linear(4096,102),
                                nn.LogSoftmax(dim=1))
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    if arch == 'densenet121':
        classifier = nn.Sequential(nn.Linear(1024, 1000),
                                nn.ReLU(),
                                nn.Dropout(0.4),
                                nn.Linear(1000,102),
                                nn.LogSoftmax(dim=1))
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    if arch == 'resnet101':
        classifier = nn.Sequential(nn.Linear(2048, 1000),
                                nn.ReLU(),
                                nn.Dropout(0.4),
                                nn.Linear(1000,102),
                                nn.LogSoftmax(dim=1))
        model.fc = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)


    criterion = nn.NLLLoss()

    device = torch.device("cuda" if (torch.cuda.is_available() == True) & (gpu == True) else "cpu")
    print(device)
    ### Train model
    model.to(device)
    every_batch =5
    iteration = 0

    print('start training')
    train_loss = []
    validation_loss = []
    train_accuracy = []
    validation_accuracy = []

    for i in range(epochs):
        running_loss = 0
        accuracy = 0
        for images, labels in trainloaders:

            images, labels = images.to(device), labels.to(device)

            iteration += 1
            print('data to device')
            optimizer.zero_grad()

            log_ps = model.forward(images)
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

        model.eval()
        with torch.no_grad():
            print('start validation')
            valid_running_loss = 0
            valid_accuracy = 0
            for valid_images,valid_labels in validloaders:
                valid_images, valid_labels = valid_images.to(device), valid_labels.to(device)
                valid_log_ps = model.forward(valid_images)
                valid_loss = criterion(valid_log_ps, valid_labels)
                valid_running_loss += valid_loss.item()

                valid_ps = torch.exp(valid_log_ps)
                valid_top_p, valid_top_class = valid_ps.topk(1, dim = 1)
                #         print(top_class)
                valid_equals = valid_top_class == valid_labels.view(*valid_top_class.shape)
                valid_accuracy += torch.mean(valid_equals.type(torch.FloatTensor)).item()
        model.train()


        print('epoch: ', i + 1)
        train_loss.append(running_loss/len(trainloaders))
        print('train loss of this epoch: ', train_loss[i])
        train_accuracy.append(accuracy/len(trainloaders))
        print('accuracy of the epoch: ', train_accuracy[i])
        validation_loss.append(valid_running_loss/len(validloaders))
        print('valid loss of this epoch: ', validation_loss[i])
        validation_accuracy.append(valid_accuracy/len(validloaders))
        print('accuracy of the valid set: ', validation_accuracy[i])
        
    fig1 = plt.figure(figsize = (10,6))
    print('train_loss', train_loss)
    plt.plot(train_loss, label = 'Training')
    plt.plot(validation_loss, label = 'Validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    fig1.savefig(os.path.join(os.getcwd(),'loss.png'))

    fig2 = plt.figure(figsize = (10,6))
    print('train_accuracy', train_accuracy)
    plt.plot(train_accuracy, label = 'Training')
    plt.plot(validation_accuracy, label = 'Validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Acurracy')
    plt.title('Training vs Validation Accuracy')
    fig2.savefig(os.path.join(os.getcwd(),'accuracy.png'))

    create_checkpoint(model, save_dir, arch, train_datasets.class_to_idx, optimizer, epochs)


def main():
    start_time = time.time()
    ### input argument
    in_args = init_argparse()
    print(in_args)

    train_model(in_args.data_dir, in_args.save_dir, in_args.learning_rate, in_args.hidden_units, in_args.epochs, in_args.gpu, in_args.batch_size, in_args.arch)
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
