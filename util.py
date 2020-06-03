# Made by Hyun Ryu, 2020/06/02
# CNN on guitar chord classification

import glob
import torch
from torch.utils import data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================================
# image processing functions
# ============================================================================
def load_jpg_data(dataFile, dtype='int'):
    """
    Loads data from a multilayer .tif file.  
    Returns result as a 3d numpy tensor.
    """

    # load the data from multi-layer TIF files
    dataImg = Image.open(dataFile)
    X = []
    while True:
        Xi = np.array(dataImg, dtype=dtype)
        X.append(Xi)
        try:
            dataImg.seek(dataImg.tell()+1)
        except EOFError:
            break

    # Put data together into a tensor.
    #
    # Also must "permute" dimensions (via the transpose function) so
    # that the slice dimension is first (for some reason, dstack seems
    # to do something odd with the dimensions).
    
    X = np.dstack(X).transpose((2,1,0))
    return X

#Leave code for debugging purposes
if __name__ == '__main__':
    mydir = "./raw_data/Emajor/test15/*.jpg"
    class_num = 5
    
    for i, file in enumerate(glob.glob(mydir)):
        img = load_jpg_data(file)
        img = torch.tensor(img).float()
        label = torch.tensor(class_num)
        
        datapair = (img, label)
        torch.save(datapair, './data_train/data/Emajor_test15_%d.pt' % (i+1))

