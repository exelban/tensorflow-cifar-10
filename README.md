# tensorflow-cifar-10
Cifar-10 convolutional network implementation example using TensorFlow library.
![](https://s3.eu-central-1.amazonaws.com/serhiy/Github_repo/Zrzut+ekranu+2017-03-19+o+19.10.46.png)

## Requirement
**Library** | **Version**
--- | ---
**Python** | **^3.5**
**Tensorflow** | **^1.6.0**
**Numpy** | **^1.12.0** 
**Pickle** |  *  

## Accuracy

## Installation

## Usage

## Training time

## Model

## What's new

### v1.0.0
    - Removed all references to cifar 100
    - Small fixes in data functions
    - Almost fully rewrited train.py
    - Simplyfy cnn model
    - Changed optimizer to AdamOptimizer
    - Changed Licence to MIT
    - Removed confusion matrix (don't like to have unnecessary dependencies)
    - Improved accuracy on testing data set (up to 79%)
    - Small fixes in train.py
    - Changed saver functions (now session will be saved only if accuracy in this session will be better than the last saved)

### v0.0.1
    - Make tests on AWS instances
    - Model fixes
    - Remove cifar-100 dataset


### v0.0.0
    - First release

## License
[Apache License 2.0](https://github.com/exelban/tensorflow-cifar-10/blob/master/LICENSE)
