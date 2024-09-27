# LeNet_MNIST
Implementation of a LeNet for Handwritten classification tasks using the MNIST dataset provided by PyTorch. The repository has 4 python programs: 

  1. ***cnn.py*** : The implementation of the LeNet in the PyTorch framework.
  2. ***process_data.py*** : Functions to pre-process data. Also works as an utils module.
  3. ***training.py*** : Training the LeNet and checks the accuracy of the model.
  4. ***testing.py*** : Testing the LeNet for a random batch of the test set (result in images folder).

The MNIST dataset consist on $$\approx 1000$$ images of handwritten digits from 0 to 9 in gray scale. For more information, [click here](https://yann.lecun.com/exdb/mnist/). 

The advantage of the *CNN* over the *MLP* is the ability to recognize patterns in the input image as the networks gets deeper due to the convolutional layers, which generally are formed by a convolution and a max pooling. The number of channels increases and the 2D dimensions decrease along the network, which final output is flattened and connected to *fully connected layers*, i.e. a MLP with SoftMax output.

The model archieves an accuracy of 99% both for training and test set with 10 epochs. For intuition purposes, the model was also trained for 1 epoch, archieving a result of 99% for training set and 98% for test set. The models were saved in the *model* folder. Also, figures for both 1 and 10 epochs models costs are available at the *figures* folder.

A picture with 40 images out of a random batch of the test set with size 64 is found at the *images* folder. If the prediction is equal to the target, the predictions displays in color blue, else, it displays in red. 

**Note**: It's interesting to see the power of the *CNN* looking at the wrong prediction.  Althought the correct target is $$5$$, the handwritten image could lead to a plausible confusion to pick $$9$$ as the answer, this is what happens to the model. It isn't mispredicting a target $$5$$ with another random number, it is mispredicting it with a number that in the picture seems very similar to a $$9$$, i.e., the *CNN* doesn't rely only on the pixel values, but on the spatial structure of the image.


