# Lab exercises: Machine Learning

<p align="center">
  <img width="300" src="./../../09DL/assets/ml_project_teaser.png">
</p>

Code: see ./09DL/


------

## Introduction
We continue the project from 09DL. If you want to run the shell from the readme, make sure to move this readme file in the right folder.


------

## Q5: Convolutional Neural Networks


Oftentimes when training a neural network, it becomes necessary to use layers more advanced than the simple Linear layers that you’ve been using. One common type of layer is a Convolutional Layer. Convolutional layers make it easier to take spatial information into account when training on multi-dimensional inputs. For example, consider the following Input:

$$
\text { Input }=\left[\begin{array}{ccccc}
x_{11} & x_{12} & x_{13} & \ldots & x_{1 n} \\
x_{21} & x_{22} & x_{23} & \ldots & x_{2 n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
x_{d 1} & x_{d 2} & x_{d 3} & \ldots & x_{d n}
\end{array}\right]
$$


If we were to use a linear layer, similar to what was done in Question 3, in order to feed this input into your neural network you would have to flatten it into the following form:

$$
\text { Input }=\left[\begin{array}{lllll}
x_{11} & x_{12} & x_{13} & \ldots & x_{1 n} \ldots x_{d n}
\end{array}\right]
$$


But in some problems, such as image classification, it's a lot easier to recognize what an image is if you are looking at the original 2-dimensional form. This is where Convolutional layers come in to play.

Rather than having a weight be a 1-dimensional vector, a 2d Convolutional layer would store a weight as a 2 d matrix:

$$
\text { Weights }=\left[\begin{array}{ll}
w_{11} & w_{12} \\
w_{21} & w_{22}
\end{array}\right]
$$


And when given some input, the layer then convolves the input matrix with the output matrix. After doing this, a Convolutional Neural Network can then make the output of a convolutional layer 1-dimensional and passes it through linear layers before returning the final output.

A 2d convolution can be defined as follows:

$$
\text { Output }=\left[\begin{array}{ccccc}
a_{11} & a_{12} & a_{13} & \ldots & a_{1 n} \\
a_{21} & a_{22} & a_{23} & \ldots & a_{2 n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
a_{d 1} & a_{d 2} & a_{d 3} & \ldots & a_{d n}
\end{array}\right]
$$

Where $a_{ij}$ is created by performing an element wise multiplication of the Weights matrix and the section of the input matrix that begins at $x_{ij}$ and has the same width and height as the Weights matrix. We then take the sum of the resulting matrix to calculate $a_{ij}$. For example, if we wanted to find $a_{22}$, we would multiply Weights by the following matrix:

$$
\left[\begin{array}{ll}
x_{22} & x_{23} \\
x_{32} & x_{33}
\end{array}\right]
$$

to get

$$
\left[\begin{array}{ll}
x_{22} * w_{11} & x_{23} * w_{12} \\
x_{32} * w_{21} & x_{33} * w_{22}
\end{array}\right]
$$

before taking the sum of this matrix $a_{22}=x_{22}∗w_{11}+x_{23}∗w_{12}+x_{32}∗w_{21}+x_{33}∗w_{22}$

Sometimes when applying a convolution, the Input matrix is padded with $0$'s to ensure that the output and input matrix can be the same size. However, in this question that is not required. As a result, your output matrix should be smaller than your input matrix.

Your task is to first fill out the Convolve function in `models.py`. This function takes in an input matrix and weight matrix, and Convolves the two. Note that it is guaranteed that the input matrix will always be larger than the weights matrix and will always be passed in one at a time, so you do not have to ensure your function can convolve multiple inputs at the same time.

After doing this, complete the DigitConvolutionalModel() class in `models.py`, the `train_digitconvolution` method in `train.py`, and the `digitconvolution_loss` in `losses.py`. You can reuse much of your code from question 3 here.

The autograder will first check your convolve function to ensure that it correctly calculates the convolution of two matrices. It will then test your model to see if it can achieve and accuracy of 80% on a greatly simplified subset MNIST dataset. Since this question is mainly concerned with the Convolve() function that you will be writing, your model should train relatively quick.

In this question, your Convolutional Network will likely run a bit slowly, this is to be expected since packages like PyTorch have optimizations that they use to speed up convolutions. However, this should not affect your final score since we provide you with an easier version of the MNIST dataset to train on.

Model Hints: We have already implemented the convolutional layer and flattened it for you. You can now treat the flattened matrix as you would a regular 1-dimensional input by passing it through linear layers. You should only need a couple of small layers in order to achieve an accuracy of 80%.

```
python autograder.py -q q5
```

------
These exercises are heavily based on the projects from [Introduction to Artificial Intelligence at UC Berkeley](https://ai.berkeley.edu/home.html).
