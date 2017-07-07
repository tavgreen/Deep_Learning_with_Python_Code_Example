# Deep Learning with Python Code Example #
## Basic Python Programming ##
All projects will be run on [Python3.6](https://www.python.org/downloads/release/python-360/), [Tensorflow](http://tensorflow.org),[Keras](http://keras.io/),[Sklearn](http://scikit-learn.org/stable/) and Matplotlib. If you are not familiar with python programming fundamental, [Tutorialspoint](http://www.tutorialspoint.com/python/) can be utililized for practising python programming. 
## Machine Learning ##
Machine learning is useful to classify or predict unstructured data. It can be applied in Speech Processing, Computer Vision and other fields. Several examples of implementation of Machine Learning are:
1. suppose we want to predict music genre given music data. Example program from [Stevetjoa](https://github.com/stevetjoa/stanford-mir) can be useful to describe how music genre can be predicted by Machine Learning.
2. suppose we want to classify land use type, whether baseball field, water, urban and so on. machine learning can be useful for classifying the type of land use. Read [Land Use Classification](https://github.com/tavgreen/landuse_classification).
3. suppose we want to improve image quality or incomplete image, machine learning can be useful for this task. Read [Image Generation](https://github.com/tavgreen/generating_images)

### Project Description ###
Deep Learning is a part of machine learning task, so the first thing should be accomplished is to understand basic of machine learning. This [book chapter 5](http://www.deeplearningbook.org) can be utilized to understand machine learning at briefly. In order to classify or predict some cases using machine learning, dataset for training data is required. in this project, we use [Movie Genre Dataset]((https://www.kaggle.com/neha1703/movie-genre-from-its-poster)) from [Kaggle](https://www.kaggle.com/) to classify genre of image movie poster. Given the movie poster of movie, we want to classify genre of that movie like Adventure, Comedy, Drama and so on. Here the step of conducting this project:
1. **Download dataset**: [Movie Genre Dataset]((https://www.kaggle.com/neha1703/movie-genre-from-its-poster)) 
2. **Feature Extraction**: Extract features from movie poster dataset. Feature is a characteristics of data in image,sound and other data that can be differentiate with other. The example are: color or shape in images, MFCC in sound and so on. There are several methods for feature extraction in images like Color Histogram, Edge Orientation, Vegetation Index and so on. in this project, we use Scale Invariant Feature Transform [SIFT](http://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html) for Feature Extraction method.
3. **Feature Selection**: Feature extraction produces vector that representing images. We don't have to use all of vector because not all vector informative to be used and can cause slowly in processing time. We have to select appropriate feature using feature selection technique. in this project, we will use [L1 Feature Selection](http://scikit-learn.org/stable/modules/feature_selection.html)
4. **Training**: Training is a part of m
5. **Performance Evaluation**:
6. **Testing New Data**:

### Program ###
**not yet finished**

### Result ###
**not yet finished**

## Deep Learning ##
The result of machine learning program above section to classify genre of movie poster is good enough for simple cases. but above program use handcrafted features that not always fit with certain large-scale data. For example we want to make an output as similar as an input (image generation). Given x = {images-1,..images-n}, we want to draw output y as similar as x, this case can be solved using Auto-Encoder (Read my previous [articles](https://github.com/tavgreen/generating_images)). another case is imagine we want to classify genre movie poster without handcrafted feature(edge orientation of image, RGB Histogram and so on) and change it into Neural Network that can make result as similar as data. In this project, we want to classify genre movie poster using Deep Neural Network with step as follows:
1. **Downloading Dataset**
2. **Defining a Model**
3. **Training Model**
4. **Evaluating Model**
5. **Testing Model**

### Program ###
Deep learning can be developed by using several tools or libraries like [Tensorflow](http://tensorflow.org), [Pytorch](http://pytorch.org) and so on. in this tutorials, we will use Tensorflow running on Python. The first step is to install environment tools like [Anaconda](https://www.continuum.io/downloads) to easily developed Python code and its libraries. Python is already available in Anaconda, so you dont have to install it anymore. Tensorflow should be install after finishing Anaconda installation by following this [Tensorflow Installation in Conda](https://anaconda.org/conda-forge/tensorflow). We will create simple Neural Network (Perceptron) as follows:

![Fig.2](https://raw.github.com/tavgreen/generating_images/master/file/formula.png?raw=true "Perceptron")

There are two input: x1 = 1 and x2 = 0, with initialization weight w1 = -0.5 and initialization w2 = 0.2. We want to compute y (output) as similar as ground truth / y_true so we need to arrange our architecture well. input or output can't be changed but we can modify value of w1 and w2 in order to make input as similar as output. let say we have x = image of cat, y1 = cat label and y2 = dog label. so our system should be make x as similar y1. 

In this case, we give **y_true = 1**. Before computing y, we have to compute h1 first. output from h1 will be activated using [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function). Later, we will use deep learning architecture that consists of more hidden like h1 to produce y as similar as y_true. Here step-by-step perceptron implementation in Tensorflow:

- Import Tensorflow library.
```python
import tensorflow as tf
```
in the code above, tensorflow is aliased by tf. so later you just called 'tf' to use tensorflow

- Define input data and output data as constant. 
```python
x1 = tf.constant(1.0,name='x1')
x2 = tf.constant(0.0,name='x2')
y_true = tf.constant(1.0,name='y_true')
```
in the code above, *tf.constant* can be used to define constant value and store it into x1, x2 and y_true.

- define weight
```python
w1 = tf.Variable(-0.5,name='w1')
w2 = tf.Variable(0.2,name='w2')
```
in the code above, *tf.Variable* can be used to define variable (can be modify) and give default value -0.5 and 0.2 respectively to w1 and w2.

- Define hidden layer.
```python
h11 = tf.multiply(x1,w1)
h12 = tf.multiply(x2,w2)
h1 = tf.add(h11,h12)
```
in the code above, we do multiplication between x1 and w1 (look the image architecture) and multiplication between x2 and w2. the result of both multiplication will be added into h1.

- Define output layer
```python
y_predict = tf.nn.sigmoid(h1)
```
in the code above, we give activation function(sigmoid) to h1. you can check [Tensorflow nn library](https://www.tensorflow.org/api_docs/python/tf/nn)  for more activationa function. The result is stored at 'out' variable.

- define loss function
```python
loss = tf.pow((y_predict - y_true),2)
```
in the code above, we define loss function using Mean Square Error (MSE). we want to know how much loss y_predict(computation in your model) to ground truth (y_true)
![Fig.3 MSE](https://wikimedia.org/api/rest_v1/media/math/render/svg/67b9ac7353c6a2710e35180238efe54faf4d9c15)

- define optimizer

is that finish? **NO, you have not run the model**. The model have just created, but you have not run the model. 


## Convolutional Neural Network ##

**not yet finished**

## References ##
- Kaggle. [Movie Poster Dataset](https://www.kaggle.com/neha1703/movie-genre-from-its-poster)
- Goodfellow et al. 2014. [Deep Learning](http://www.deeplearningbook.org)
- Lecun et al. 2015. [Deep Learning](https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)
- CS231N Stanford University.[Convolutional Neural Network](http://cs231n.github.io)
- Quoc V.Le. [Tutorial in Deep Learning](https://cs.stanford.edu/~quocle)

