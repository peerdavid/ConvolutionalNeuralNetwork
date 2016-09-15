# Convolutional Neural Networks
This project trains and evaluates convolutional neural networks with tensorflow.
Images (png or jpeg) will be loaded from your file system. To create your
own training / evaluation set, the tool **datr** (https://github.com/peerdavid/datr) can be used.
Simply download images with datr and put them into folders named 0, 1, ... for all different classes
and set the folder in the params.py (FLAGS.img_dir) file. This images can be used, to train your
cnn using the following components:

## params.py
This file contains all flags which can be set such as learning_rate, img_dir, evaluation_size etc.


## data_input.py
Is responsible for loading training, evaluation and ing and images for predictions.
Training, testing and evaluation data must be structured into the following folders:<br>
```
img_dir_train
   |--0
   |  |-###.jpg
   |  |-###.jpg
   |
   |--1
   |  |-###.jpg

img_dir_test
   |--0
   |  |-###.jpg
   |  |-###.jpg
   |
   |--1
   |  |-###.jpg
```

The name of the folder is the label used for the training.


## train.py
Call python3 train.py to train your cnn (which is defined in model.py). 
With the FLAGS.optimizer flag you can define your favorite optimizer.
After 10 steps, a console log will output the current loss value, after 100 steps
general values for tensorboard are written into the log_dir. After 1000 steps 
the training and evaluation (evaluation data will only be used for evaluation, never for training) accuracy
will be calculated and written into tensorboard. After 2000 steps a checkpoint file will be written
into the log_dir. This files will be used by evaluation.py and predict.py
<br><br>
To see the accuracy, conv1 kernel, histograms, loss etc. during the training, start tensorboard and
open http://localhost:6006/ in your browser.
```
tensorboard --logdir path/to/log_dir
```

## model.py
This file contains the architecture for your convolutional neural network.
The example contains the following architecture:
INPUT -> [CONV -> POOL -> NORM]*2 -> FC -> FC -> SOFTMAX

The kernel of the conv1 layer will also be displayed in tensorboard:
<img src="documentation/conv1_kernel.png" alt="conv1_kernel"/>


## evaluation.py
To evaluate different network architectures after the training, you can use evaluation.py
Set the eval_dir and call python3 evaluation.py (Note: The structure of the directory must be the same as the img_dir).
The images used in eval_dir should not be contained in your training data. evaluation.py will calculate the accuracy
and the confusion matrix. Example: <br>
```
Restore session from checkpoint log/model.ckpt-4001              
                                                                 
Running evaluation for data/mnist_test/

Top-1-Accuracy | Num examples: 10000  Num correct: 9793  Precision @ 1: 0.9793
Top-2-Accuracy | Num examples: 10000  Num correct: 9956  Precision @ 1: 0.9956

[[ 976    0    0    0    0    1    7    0    2    0]
 [   0 1133    4    1    0    0    5    3    1    1]
 [   2    1 1023    3    1    0    0    6    6    1]
 [   0    0    0 1002    0    8    0    0   25    2]
 [   0    0    0    0  964    0    7    0    3    2]
 [   1    0    0    1    0  881   20    1   27    3]
 [   0    0    0    0    0    0  918    0    1    0]
 [   0    1    5    2    1    0    0 1012    3    3]
 [   1    0    0    0    0    0    1    0  887    0]
 [   0    0    0    1   16    2    0    6   19  997]]
 
 
```

## predict.py
If you want to predict images from your trained model, set the FLAGS.checkpoint in params.py
and all images you want to predict inside the predict.py file. The output will be the predicted
class of your (already trained) network.