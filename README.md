# tensorflow-cifar-10
Cifar-10 convolutional network implementation example using TensorFlow library.
![](http://bodya.net/data/Zrzut%20ekranu%202017-03-19%20o%2019.10.46.png)

## Requirement
**Library** | **Version**
--- | ---
**Python** | **^3.5**
**Tensorflow** | **^1.0.1** 
**Numpy** | **^1.12.0** 
**Pickle** |  *  

## Usage
### Download code:
```sh
git clone https://github.com/exelban/tensorflow-cifar-10

cd tensorflow-cifar-10
```

### Train cnn:
Batch size: 128

After every 1000 iteration making prediction on all testing batch. 

```sh
python3 train.py
```
Example output:
```sh
Trying to restore last checkpoint ...
Restored checkpoint from: ./tensorboard/cifar-10/-20000
Global Step:  20010, accuracy:  98.4%, loss = 0.04 (1381.6 examples/sec, 0.09 sec/batch)
Global Step:  20020, accuracy:  99.2%, loss = 0.02 (1370.4 examples/sec, 0.09 sec/batch)
Global Step:  20030, accuracy: 100.0%, loss = 0.01 (1375.0 examples/sec, 0.09 sec/batch)
Global Step:  20040, accuracy:  98.4%, loss = 0.04 (1401.3 examples/sec, 0.09 sec/batch)
Global Step:  20050, accuracy: 100.0%, loss = 0.01 (1358.1 examples/sec, 0.09 sec/batch)
Global Step:  20060, accuracy: 100.0%, loss = 0.02 (1289.0 examples/sec, 0.10 sec/batch)
Global Step:  20070, accuracy: 100.0%, loss = 0.01 (1305.6 examples/sec, 0.10 sec/batch)
Global Step:  20080, accuracy:  98.4%, loss = 0.05 (1421.1 examples/sec, 0.09 sec/batch)
Global Step:  20090, accuracy:  99.2%, loss = 0.01 (1411.4 examples/sec, 0.09 sec/batch)
Global Step:  20100, accuracy: 100.0%, loss = 0.00 (1369.6 examples/sec, 0.09 sec/batch)
Accuracy on Test-Set: 76.23% (7623 / 10000)
Saved checkpoint.
```

#### Make prediction:
```sh
python3 predict.py
```

## Model

| ** Layers ** |
| :---: |
| ** Convolution layer 1 ** |
| ** ReLu 1 ** |
| ** MaxPool 1 ** |
| ** Normalization 1 ** |
| --- |
| ** Convolution layer 2 ** |
| ** ReLu 2 ** |
| ** Normalization 2 ** |
| ** MaxPool 2 ** |
| --- |
| ** Convolution layer 3 ** |
| ** ReLu 3 ** |
| ** Convolution layer 4 ** |
| ** ReLu 4 ** |
| ** Convolution layer 5 ** |
| ** ReLu 5 ** |
| ** Normalization 3 ** |
| ** MaxPool 3 ** |
| --- |
| ** Fully connected 1 ** |
| ** Fully connected 2 ** |
| ** Softmax_linear ** |

![](http://bodya.net/data/Zrzut%20ekranu%202017-03-19%20o%2019.11.18.png)
