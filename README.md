# MNIST Neural Network Testing

This code was developed as part of a project for an Artificial Intelligence Class. It utilizes a neural network to identify handwritten characters.

# Documentation

The current capabilities of this project is to identify the hand-written digits 0-9. The neural network can be trained on both MNIST data and real-world data.

## Command Line Arguments

The following command line arguments are available:

* **-v:** Enables verbose output
* **-ev:** Enables extreme verbose output. this will cause the network to print what the neural network is predicting and what the actual value is.
* **-g:** Enables graph output. The application has the ability to produce graphs using pyplot.
* **-t:** Allows us to specify a topology for the neural network. For example, *-t 784,500,10*. This would create a network with and input layer of 784 nodes, a hidden layer with 500 nodes and an output layer with 10 nodes.
* **-n:** Specifies a name for the neural network. Currently this is only used for the title of graphs.
* **-e:** Specifies the number of epochs for the network.
* **-train:** Allows us to specify a path to a set of images to use for taining the network. If this isn't specified, the network will train off the MNIST dataset.
* **-test:** Allows us to specify a path to a set of images to test with the neural network.

## License

Copyright 2018 Adam Thompson <adam@serialphotog.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
