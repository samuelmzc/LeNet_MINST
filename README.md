# LeNet_MINST
Implementation of a LeNet for Handwritten classification tasks using the MNIST dataset provided by PyTorch. The repository has 4 python programs: 

  1. ***cnn.py*** : The implementation of the LeNet in the PyTorch framework.
  2. ***process_data.py*** : Functions to pre-process data. Also works as a utils module.
  3. ***training.py*** : Training the LeNet and checks the accuracy of the model.
  4. ***testing.py*** : Testing the LeNet for a random batch of the test set (result in images folder).

The MNIST dataset consist on $$\approx 1000$$ images of handwritten digits from 0 to 9 in gray scale. For more information, [click here](https://yann.lecun.com/exdb/mnist/). 

The advantage of the *CNN* over the *MLP* is the ability to recognize patterns in the input image as the networks gets deeper due to the convolutional layers, which generally are formed by a convolution and a max pooling. The number of channels increases and the 2D dimensions decrease along the network, which final output is flattened and connected to *fully connected layers*, i.e. a MLP with SoftMax output.


