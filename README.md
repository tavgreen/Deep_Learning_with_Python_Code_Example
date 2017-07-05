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
not yet finished

### Result ###
not yet finished

## Deep Learning ##
The result of machine learning program above section to classify genre of movie poster is good enough for simple cases. but above program use handcrafted features that not always fit with certain large-scale data. For example we want to make an output as similar as an input (image generation). Given x = {images-1,..images-n}, we want to draw output y as similar as x, this case can be solved using Auto-Encoder (Read my previous [articles](https://github.com/tavgreen/generating_images)). another case is imagine we want to classify genre movie poster without handcrafted feature(edge orientation of image, RGB Histogram and so on) and change it into Neural Network that can make result as similar as data. In this project, we want to classify genre movie poster using Deep Neural Network with step as follows:
1. **Downloading Dataset**
2. **Defining a Model**
3. **Training Model**
4. **Evaluating Model**
5. **Testing Model**

To be continue...

## Convolutional Neural Network ##
on progress

## References ##
- Kaggle. [Movie Poster Dataset](https://www.kaggle.com/neha1703/movie-genre-from-its-poster)
- Goodfellow et al. 2014. [Deep Learning](http://www.deeplearningbook.org)
- Lecun et al. 2015. [Deep Learning](https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)
- CS231N Stanford University.[Convolutional Neural Network](http://cs231n.github.io)
- Quoc V.Le. [Tutorial in Deep Learning](https://cs.stanford.edu/~quocle)

