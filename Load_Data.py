from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

### process individual image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    width, height = im.size
    
    aspect_ratio = width / height
    if width < height:
        new_width = 256
        new_height = int(new_width / aspect_ratio)
    elif height < width:
        new_height = 256
        new_width = int(width * aspect_ratio)
    else: # when both sides are equal
        new_width = 256
        new_height = 256
    im = im.resize((new_width, new_height))
    
#     im.thumbnail((256,256))
    
    
    im = im.crop((0,0,224,224))
#     print(im)
    
    ##im.show()

    np_image = np.array(im)
#     print(np_image.shape)
#     print(np_image.shape)
#     print(np_image)
    ### normalization 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image/255 - mean)/std
#     print(nm_image.shape)    
#     
    image_T =np_image.transpose((2,0,1))
    ##
    ##image_T = np_image
    image_T = torch.tensor(image_T)
#     print(image_T.shape)
#     print(image_T)
    ##print(np_image/255)
    return image_T


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax