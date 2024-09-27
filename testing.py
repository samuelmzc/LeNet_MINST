import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch as torch
import torchvision as tv
from cnn import *
from process_data import *

SIZE_IMG = 32
BATCH_SIZE = 64
input_shape = (1, 32, 32)
input_channels = input_shape[0]
n_classes = 10

model_path = "model/LeNet_MNIST_weights1ep.pth"

CONVnet = CNN(input_channels, n_classes)

print("Loading model.....")
CONVnet.load_state_dict(torch.load(model_path, weights_only = True))
print("Model loaded!")

train_set, test_set = load_process_data(size = SIZE_IMG, num_classes = n_classes)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = True)

# Predicting image
im = Image.open("images/test.jpeg").convert("L")
im2 = Image.open("images/tes2.jpeg").convert("L")
im = ImageOps.invert(im)
im2 = ImageOps.invert(im2)

data_transformations = tv.transforms.Compose([
        tv.transforms.Resize((SIZE_IMG, SIZE_IMG)),                # Resize image
        tv.transforms.ToTensor(),                        # Converts to torch tensor and normalize to [0, 1]
        tv.transforms.Lambda(lambda img : (img * 2) - 1) # From [0, 1] to [-1, 1]
    ])

tensortest1 = data_transformations(im)
tensortest2 = data_transformations(im2)

try_loader = torch.utils.data.DataLoader([tensortest1, tensortest2])

tensor_im_test = next(iter(try_loader))

print(tensor_im_test[0])



# Predicting labels
tensors, targets = next(iter(test_loader))
predictions = CONVnet.forward(tensors)
_ , scores = predictions.max(1)

n_rows = 4
n_cols = 10
fig, axs = plt.subplots(n_rows, n_cols)

for i in range(n_rows):
    for j in range(n_cols):
        total_idx = n_cols * i + j
        axs[i, j].xaxis.set_tick_params(labelbottom=False) # Remove x ticks
        axs[i, j].yaxis.set_tick_params(labelleft=False)   # Remove y ticks
        image = tensor_to_PIL(tensors[total_idx])
        axs[i, j].set_title(f"Target: {targets[total_idx]}")
        axs[i, j].set_xlabel(f"Prediction: {scores[total_idx]}")
        if targets[total_idx] == scores[total_idx]:
            axs[i, j].xaxis.label.set_color("blue")
        else:
            axs[i, j].xaxis.label.set_color("red")
        axs[i, j].imshow(image, cmap = "gray")

plt.show()


