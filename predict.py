# Basic usage: python predict.py /path/to/image checkpoint
# Options:
# Return top KK most likely classes: python predict.py input checkpoint --top_k 3
# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# Use GPU for inference: python predict.py input checkpoint --gpu


import torchvision.models as models
import torch
from torch import optim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Load_Data import process_image, imshow
import argparse
import json


def init_argparse():
    parser = argparse.ArgumentParser(description = '')
    ###
    parser.add_argument('--image_direction', type = str,
                        help = 'input image file directory')
    ### # Argument : Choose architecture
    parser.add_argument('--arch', action = 'store', dest = 'arch', type = str, default = 'vgg16_bn')
    parser.add_argument('--checkpoint',action ='store', dest = checkpoint, type = str,
                        help = 'pretrained model file')
    parser.add_argument('--topk',action = 'store', dest = topk, type = int, default = 3,
                        help = 'number of top k class presented in the output')
    parser.add_argument('--category_names',action = 'store', dest = 'category_names', type = str, default = './cat_to_name.json',
                        help = 'mapping of the category name')
    parser.add_argument('--gpu', action = 'store_true',dest = 'gpu', default = True,
                        help = '')
    # img_path = './flowers/train/43/image_02338.jpg'

    args = parser.parse_args()
    return args

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

device = torch.device('cuda' if (torch.cuda.is_available() == True) & (in_args.gpu == True) else 'cpu' )

# # TODO: Write a function that loads a checkpoint and rebuilds the model
# def load_checkpoint(filepath, arch):
#     checkpoint = torch.load(filepath)
#     ### initialize model and optimizer
#     model = getattr(models, arch)(pretrained = True)
#     model.classifier = checkpoint['classifier']
#     model.load_state_dict(checkpoint['state_dict'])
#
#     optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epochs = checkpoint['epochs']
#
#     return model

def load_model(checkpoint):

    t_models = {
                'vgg19' : models.vgg19(pretrained = True),
                'densenet121' : models.densenet121(pretrained = True),
                'resnet101' : models.resnet101(pretrained = True)}

    model = t_models.get(checkpoint['arch'], 'vgg19')

    if checkpoint['arch'] == 'vgg19' or checkpoint['arch'] == 'densenet121':
        model.classifer = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']

    else:
        model.fc = checkpoint['fc']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']

    return model

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    ## load image
    image = process_image(image_path)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])).type(torch.FloatTensor)
    image = image.to(device)
    model.to(device)
    ## predict
    ps_log = model.forward(image)
    ps = torch.exp(ps_log)

    top_p, top_class = ps.topk(topk, dim = 1)

    return top_p, top_class

# TODO: Display an image along with the top 5 classes
# img_path = './flowers/train/43/image_02338.jpg'

def main():
    start_time = time.time()
    ### input argument
    in_args = init_argparse()
    print(in_args)
    nn_model = load_model(checkpoint)
    nn_model.eval()

    top_p, top_class = predict(image_direction, nn_model, topk)
    print('top class: ', [cat_to_name[str(i)] for i in top_class.tolist()[0]])
    print('top possibility: ', top_p.tolist()[0])

    end_time = time.time()

    # TODO 0: Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time #calculate difference between end time and start time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
# plt.barh([cat_to_name[str(i)] for i in top_class.tolist()[0]], width = top_p.tolist()[0])
# plt.show()
# imshow(img_path)

if __name__ = '__main__':
    main()
