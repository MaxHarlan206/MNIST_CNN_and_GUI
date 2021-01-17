# MNIST_CNN_and_GUI
Solving the MNIST dataset with a convolution neural network and allowing custom input with a GUI

Purpose:<br>
I am in the process of making this project to get a strong foundation in convolutional neural networks, image recognition, and gui creation (three fields which I have had almost no experience in before).

Features:<br>
This project currently takes 28x28 px hand drawn images of numbers from the MNIST database of handwritten digits which look like this:
![](https://raw.githubusercontent.com/MaxHarlan206/MNIST_CNN_and_GUI/main/Screenshot%202021-01-16%2022%3A00%3A18.png)
<br>... and uses a convolutional neural net to predict with 99%+ accuracy which digit is pictured in this format.
<br>
The layers of the network are:
- Image normalization
- A convlutional layer
- A max pooling layer
- A flattening layer 
- Two dense layers 

Using the best trained model from this cnn, users can draw digits of their choice into a gui which uses primarily tkinter to select and predict numbers:<br>
![](https://raw.githubusercontent.com/MaxHarlan206/MNIST_CNN_and_GUI/main/MNIST.gif)

Right now this project is at 99.4% accuracy and the GUI version works nearer 90% because of the inconsistency of that data and the training set.
My goal is to get this accuracy up past 99.6% and make the gui version 95%. 

I owe a huge thanks to Jason Brownlee PhD of machine learning mastery as well as Jaideep Singh.
Most of this code is a hack job between resources I picked up from their aricles. 

Made with:
Numpy, Keras, Tkinter, Tensorflow, PIL, Opencv, Matplotlib, and Python3
