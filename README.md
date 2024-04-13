# Basic Linear Regression Using Numpy and Math
![Static Badge](https://img.shields.io/badge/-numpy-teal?logo=numpy)
![Static Badge](https://img.shields.io/badge/-Jupyter_Notebook-orange?logo=jupyter&logoColor=white)

## Basic Notations

### Hyperparameters

Basically parameters related to our model.
such as:

- number of layers in a nn
- neurons in each layer
- learning rate
- regularization

### Loss Function

A way to map the performance of our model into a real number. It measures how well the model is performing its task.
Its Important in learning because its what guides the update of the parameters so that the model can perform better

### The data

Its usually a good idea to partition the data in 3 different sets

**Train**: The set thats used to actually learn the mdoel. The data is presented to the model and the learning method produces a fit. aka a math function

**Validation**: The set that is used to tune the hyperparameters

**Test** : Set that is used to evaluvate the overall performance of the model

## Error Function

$$
\mathcal{L}(\hat{y}, y) = \frac{1}{M} \sum_{i=1}^{M} (\hat{y}_i - y_i)^2
$$

In order to estimate the quality of our model we need a function of error. One such function is called Squared Loss.

Mean Squared Error (MSE) would be the sum of the square of the errors for each training point, divided by the total amount of points.

Incorporating our model (y = mx + b) we obtain::

$$
\mathcal{L}(\hat{y},x,w) = \frac{1}{M}\sum_{i=1}^{M}(\hat{y}-(w^T x_i +b))^2
$$

## Gradient Descent

One of the methods we can use to minimuze Error is using Gradient Descent. 

We use gradients to update the model parameters (w and b in this case) until a minimum is found.

![sgd.gif](https://prod-files-secure.s3.us-west-2.amazonaws.com/b2d0552a-437e-4bb3-8904-b3b588bb0ac2/20c13d71-fc13-4187-9f57-10e9475969f9/sgd.gif)

June 1, 2024 

> Figure out multivariable calculus
> 

---

# The Code

```python
import numpy as np

data_x = np.linspace(1.0, 10.0, 100)[:, np.newaxis]
data_y = np.sin(data_x) + 0.1*np-power(data_x,2) + 0.5*p. random. rand (100,1) 
data_x /= np.max (data_x)
```

np.linspace(1.0, 10.0, 100) This is a function call to np.linspace(), which generates an array of 100 evenly spaced values between 1.0 and 10.0. The resulting array is one-dimensional.

[:, np.newaxis]: This is a slice operation that adds an extra dimension to the array. The : operator means "select all elements", and np.newaxis is a special index that adds an extra dimension. This is useful when you need to perform operations that require a certain number of dimensions.

So, data_x is a two-dimensional NumPy array with 100 rows and 1 column, where the values in the first column range from 1.0 to 10.0, evenly spaced.



In order to simplify our model we use a trick which consists in including the intercept in the input values, this way we dont have to carry the bias (b) term through the calculation, that’s done by adding a column of ones to the data.

> Basically subtract the intercept (b)
> 

Our model becomes:

$$
y = w^Tx
$$

```python
data_x = np.hstack((np.ones_like(data_x), data_x))
```
