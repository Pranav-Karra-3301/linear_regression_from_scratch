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

![sgd.gif](https://doimages.nyc3.cdn.digitaloceanspaces.com/010AI-ML/content/images/2018/05/68747470733a2f2f707669676965722e6769746875622e696f2f6d656469612f696d672f70617274312f6772616469656e745f64657363656e742e676966.gif)


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

## Implementing Linear Regression

Now that we have our data prepared, let's implement the linear regression algorithm using gradient descent.

### Initializing Parameters

We'll start by initializing our weight vector `w` randomly:

```python
w = np.random.randn(2, 1)
```

### Defining the Model

Our linear regression model is simply the dot product of the input `x` and the weight vector `w`:

```python
def model(X, w):
    return np.dot(X, w)
```

### Implementing the Loss Function

We'll use Mean Squared Error (MSE) as our loss function:

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
```

### Gradient Descent

Now, let's implement the gradient descent algorithm:

```python
def gradient_descent(X, y, w, learning_rate, n_iterations):
    m = len(y)
    for _ in range(n_iterations):
        y_pred = model(X, w)
        gradient = (1/m) * np.dot(X.T, (y_pred - y))
        w -= learning_rate * gradient
    return w
```

### Training the Model

Let's train our model using the gradient descent algorithm:

```python
learning_rate = 0.01
n_iterations = 1000

w_trained = gradient_descent(data_x, data_y, w, learning_rate, n_iterations)
```

### Making Predictions

Now that we have trained our model, we can use it to make predictions:

```python
y_pred = model(data_x, w_trained)
```

### Evaluating the Model

Let's calculate the Mean Squared Error to evaluate our model's performance:

```python
mse = mse_loss(data_y, y_pred)
print(f"Mean Squared Error: {mse}")
```

## Visualizing the Results

To better understand how well our model performs, let's visualize the results:

```python
import matplotlib.pyplot as plt

plt.scatter(data_x[:, 1], data_y, color='b', label='Actual data')
plt.plot(data_x[:, 1], y_pred, color='r', label='Predicted line')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Results')
plt.show()
```

This will create a scatter plot of the original data points and overlay the predicted line from our linear regression model.


Remember that while this implementation is educational, for real-world applications, you might want to use more robust libraries like scikit-learn, which offer optimized implementations and additional features.
