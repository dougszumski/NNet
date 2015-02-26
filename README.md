# NNet

[![Build Status](https://travis-ci.org/dougszumski/NNet.svg?branch=master)](https://travis-ci.org/dougszumski/NNet)

This is a C implementation of the neural network for handwriting recognition outlined in the [free online book by Michael Nielsen.](http://neuralnetworksanddeeplearning.com/)

It was written as a learning exercise, and is essentially a port of [Michael's Python implementation.](https://github.com/mnielsen/neural-networks-and-deep-learning)

## Build instructions

* Install the GNU Scientic library, GCC, G++

* Download the [training data](http://yann.lecun.com/exdb/mnist/):

```bash
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
```

* Unzip the data into ./dat:

```bash
gunzip -c train-images-idx3-ubyte.gz > dat/train-images-idx3-ubyte
gunzip -c train-labels-idx1-ubyte.gz > dat/train-labels-idx1-ubyte
```

* Build using cmake eg. from the project directory:
    * `cd build`
    * `cmake ..`
    * `make`

* Run from the project folder:
    * Tests with `./tests`
    * Train the network with `./run`

* Read the book!
