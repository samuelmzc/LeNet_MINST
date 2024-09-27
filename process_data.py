import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import torchvision as tv

def load_process_data(size, num_classes):
    """
    Loads thet training and test sets for CIFAR10 dataset

    Arguments:
    size -- int, one dimentional size of the image, for resizing purposes

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


def max_value(tensor):
    return int((tensor == torch.max(tensor)).nonzero(as_tuple = True)[0])


def plot_loss(epochs, hist, hist_t, plot):
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

def predicted_tensor(model, images):
    pred = torch.nn.Softmax(dim = 1)(model.feed_forward(images))

    for i in range(len(images)):
        pred[i] = max_value(pred[i])
    
    return pred[:, 0].to(torch.int)


def check_accuracy(model, loader):

    accuracy = 0
    samples = 0

    with torch.no_grad():
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
    hours = int(time/3600)
    minutes = int(time/60 - hours*60)
    seconds = int(time - hours*3600 - minutes*60)
    return [hours, minutes, seconds]