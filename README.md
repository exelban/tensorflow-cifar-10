# tensorflow-cifar-10
Cifar-10 convolutional network implementation example using TensorFlow library.
![](https://s3.eu-central-1.amazonaws.com/serhiy/Github_repo/Zrzut+ekranu+2017-03-19+o+19.10.46.png)

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

After every 1000 iteration making prediction on testing batch. 

20000 iteration take about 2h on GTX 960.

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

Example output:
```sh
Trying to restore last checkpoint ...
Restored checkpoint from: ./tensorboard/cifar-10/-20000
Accuracy on Test-Set: 75.73% (7573 / 10000)
[848   9  42  12  16   3   8   8  38  16] (0) airplane
[ 21 841   7   6   1   8   5   1  35  75] (1) automobile
[ 55   2 720  47  78  29  26  26   6  11] (2) bird
[ 33  10  83 587  74 118  47  24   8  16] (3) cat
[ 18   0  89  56 755  16  18  40   7   1] (4) deer
[ 18   5  77 194  58 581  15  40   4   8] (5) dog
[ 15   4  65  69  39  18 771   6   8   5] (6) frog
[ 23   0  36  36  75  30   3 789   1   7] (7) horse
[ 61  18  10   9   8   6   6   2 858  22] (8) ship
[ 41  70  10  14   3   4   2   6  27 823] (9) truck
 (0) (1) (2) (3) (4) (5) (6) (7) (8) (9)
```

## Tensorboard
```sh
tensorboard --logdir=./tensorboard
```

## Model

| **Convolution layer 1** |
| :---: |
| Conv_2d |
| ReLu |
| MaxPool |
| LRN |
| **Convolution layer 2** |
| Conv_2d |
| ReLu |
| LRN |
| MaxPool |
| **Convolution layer 3**  |
| Conv_2d |
| ReLu |
| **Convolution layer 4** |
| Conv_2d |
| ReLu |
| **Convolution layer 5** |
| Conv_2d |
| ReLu |
| LRN |
| MaxPool |
| **Fully connected 1** |
| **Fully connected 2** |
| **Softmax_linear** |
![](https://s3.eu-central-1.amazonaws.com/serhiy/Github_repo/Zrzut+ekranu+2017-03-19+o+19.11.18.png)

## License
[Apache License 2.0](https://github.com/exelban/tensorflow-cifar-10/blob/master/LICENSE)
