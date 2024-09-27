import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import torchvision as tv
import torchsummary as tsum
import time
from cnn import *
from process_data import *

# Hyperparameters
SIZE_IMG = int(32)
BATCH_SIZE = 64
EPOCHS = 1
LR = 0.001

# Number of classes to predict
n_classes = 10

# Downloading and preprocessing the data
print("\nVerifying the dataset is downloaded.... \n")

train_set, test_set = load_process_data(size = SIZE_IMG, num_classes = n_classes)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = True)

m_train = len(train_loader)
m_test = len(test_loader)

print(f"Train set size: {m_train} | Test set size: {m_test}")


# Plotting random data and targets
random_trainload_tensor, target = next(iter(train_loader))

random_train_image = tensor_to_PIL(random_trainload_tensor)

print("Showing a random sample of the training test and its target.....")
plt.title(f"Target: {str(target.numpy()[0])}")
plt.imshow(random_train_image, cmap = "binary")
plt.show()

# Building the model
input_shape = (1, 32, 32)
input_channels = input_shape[0]

model = CNN(input_channels, n_classes)
CrossEntropy = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)

print("\nSummary of the model: ")
tsum.summary(model, input_shape)


# Starting training
hist = []
hist_test = []

print("\nTraining......")
start_time = time.time()

for epoch in range(EPOCHS):

    epoch_cost = 0
    ec_test = 0

    # Nota: enumerate(xxxx_loader) tiene la siguiente estructura: nº batch, inputs, targets
    for i, (data, target) in enumerate(train_loader):
        output = model.forward(data)
        loss = CrossEntropy(output, target)
        epoch_cost += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    for i, (data, target) in enumerate(test_loader):
        output_t = model.forward(data)
        loss_t = CrossEntropy(output_t, target)
        ec_test += loss_t
        
    epoch_cost /= m_train
    ec_test /= m_test

    hist.append(float(epoch_cost))
    hist_test.append(float(ec_test))

    if (epoch + 1)%1 == 0:
        print(f"Epoch {epoch + 1} ==> Train cost : {epoch_cost : .5f} | Test cost : {ec_test : .5f}")

end_time = time.time() - start_time

hours, minutes, seconds = seconds_to_hhmmss(end_time)


if hours >= 1:
    print(f"Training complete! It took {hours} hours, {minutes} minutes and {seconds} seconds. \n")
elif minutes >= 1:
    print(f"Training complete! It took {minutes} minutes and {seconds} seconds. \n")
else:
    print(f"Training complete! It took {seconds} seconds. \n")


# Save the model

name = "LeNet_MNIST_weights1ep.pth"
path = "model/"
pathname = path + name

print("Saving the model.....")
torch.save(model.state_dict(), pathname)
print("Weights of the model saved as " + name)


# Plotting (if required) the cost function (Cross Entropy)
p = input("Show cost plot? [Y/n]: ")

if p.lower() == "y":
    print("Showing cost plot.....")
    plot_loss(epochs = EPOCHS, hist = hist, hist_t = hist_test, plot = "Y")

else:
    plot_loss(epochs = EPOCHS, hist = hist, hist_t = hist_test, plot = "N")

# Checking accuracy of the model
print("Checking accuracy......")
check_accuracy(model, train_loader)
check_accuracy(model, test_loader)