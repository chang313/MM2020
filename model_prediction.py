from model import CNN
from PIL import Image
import torch
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

model = CNN()
model.load_state_dict(torch.load("CNN_0605_epoch_30.pth"))
#for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
model.eval()
example = torch.rand(1,3,720,1280)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")



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
            dataImg.seek(dataImg.tell() + 1)
        except EOFError:
            break

    # Put data together into a tensor.
    #
    # Also must "permute" dimensions (via the transpose function) so
    # that the slice dimension is first (for some reason, dstack seems
    # to do something odd with the dimensions).

    X = np.dstack(X).transpose((2, 1, 0))
    return X

img = load_jpg_data("./data/Emajor/test15/test11-1.jpg")
x = torch.tensor(img).float()
#x=torch.load('./data/Emajor_test15_1.pt')

x.unsqueeze_(0)
print(x.shape)


y_pred = model(x)
print(y_pred.data)
_, predicted = torch.max(y_pred.data, 1)
print(predicted)