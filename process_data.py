import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import torchvision as tv

def load_process_data(size, num_classes):
    """
    Loads thet training and test sets for MNIST dataset

    Arguments:
    size -- int, one dimentional size of the image, for resizing purposes
    num_classes -- int, number of classes of the dataset

    Returns:
    train -- torch.dataset, splitted training set
    test -- torch.dataset, splitted test set
    """

    data_transformations = tv.transforms.Compose([
        tv.transforms.Resize((size, size)),                # Resize image
        tv.transforms.ToTensor(),                        # Converts to torch tensor and normalize to [0, 1]
        tv.transforms.Lambda(lambda img : (img * 2) - 1) # From [0, 1] to [-1, 1]
    ])

    target_transforms = tv.transforms.Compose([
        tv.transforms.Lambda(lambda target : torch.tensor([target])),
        tv.transforms.Lambda(lambda target : torch.nn.functional.one_hot(target, num_classes))
    ])
    

    train = tv.datasets.MNIST(root = ".", train = True, transform = data_transformations, download = True)
    test = tv.datasets.MNIST(root = ".", train = False, transform = data_transformations, download = True)

    return train, test


def tensor_to_PIL(tensor):
    """
    Apply the reverse transformations of load_process_data() to convert the tensor in a PIL image

    Arguments:
    tensor -- torch.tensor -- input tensor

    Returns:
    image -- PIL, output image
    """

    reversed_transformations = tv.transforms.Compose([
        tv.transforms.Lambda(lambda img : (img + 1) / 2), # Returns to [0, 1]
        tv.transforms.Lambda(lambda img : img.permute(1, 2, 0)), # from CHW to HWC
        tv.transforms.Lambda(lambda img : img*255),
        tv.transforms.Lambda(lambda img : img.numpy().astype(np.uint8)),
        tv.transforms.ToPILImage()
    ])

    if len(tensor.shape) == 4:
        tensor = tensor[0, :, :, :]

    image = reversed_transformations(tensor)
    return image
    

def plot_loss(epochs, hist, hist_t, plot):
    """
    Plot the cost of the training and test set vs epochs

    Arguments:
    epochs -- int, number of epochs the model is trained
    hist -- list, list with values of training cost for each epoch
    hist_t -- list, list with value of test cost for each epoch
    plot -- str, [Y,n] indicating if the user wants to show the plot
    """
    
    epocharr = np.linspace(1, epochs, epochs)
    plt.title("Cost for LeNet CNN | MNIST")
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy")
    plt.plot(epocharr, np.array(hist), label = "Train cost")
    plt.plot(epocharr, np.array(hist_t), label = "Test cost")


    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"figures/cost_{epochs}ep.pdf")
    
    if plot == "Y":
        plt.show()
    else:
        pass


def check_accuracy(model, loader):
    """
    Prints the accuracy of the model for certain data loader (train or test)

    Arguments:
    model -- class, LeNet model
    loader -- torch.utils.data.DataLoader, train/test data loader
    """

    accuracy = 0
    samples = 0

    with torch.no_grad(): # 
        for i, (data, targets) in enumerate(loader):
            output = model.forward(data)
            _, prediction = output.max(1)
            accuracy += (prediction == targets).sum()
            samples += prediction.size(0)
    
    if loader.dataset.train:
        print(f"Training set accuracy : {accuracy/samples * 100 : .0f}%")
    else:
        print(f"Test set accuracy   : {accuracy/samples * 100 : .0f}%")

def seconds_to_hhmmss(time):
    """
    Converts given amount of time (in seconds) and to an array [hours, minutes, seconds]

    Arguments:
    time -- float, time in seconds

    Returns:
    [hours, minutes, seconds] -- list, converted time to hours, minutes and seconds
    """
    
    hours = int(time/3600)
    minutes = int(time/60 - hours*60)
    seconds = int(time - hours*3600 - minutes*60)
    return [hours, minutes, seconds]
