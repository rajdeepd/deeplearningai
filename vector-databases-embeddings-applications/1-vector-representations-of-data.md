---
layout: default
title: 1. Vector Representation of Data
nav_order: 2
description: ".."
has_children: false
parent:  Vector Databases
---

## Introduction

In this lesson, you'll learn where the vectors in vector databases come from. You'll start off by looking at how neural networks can be used to represent and embed data as numbers, but also you'll be building an autoencoder architecture to embed images into vectors. You'll then go over what it means for data objects to be similar or dissimilar, and how to quantify this using the vector representations of data. Let's get into it. 


## Embeddings for MNIST Dataset

Here's the autoencoder, and to illustrate how it works, we will use the MNIST handwritten digits dataset, which will then, if you pass a digit, an image of a digit like this, which by the way has 28 by 28 pixels in dimension, which makes it for 784 dimensions, and then if you run it through this, the encoder will compress it, and then decoder will decompress it. 

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-27_at_7.36.17 PM.png" width="80%" />

We'll end up with another image. And you can already see that the two images, they don't exactly match. 

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-27_at_7.36.47 PM.png" width="80%" />

This is why we have to run this through multiple training sets. Each time we run it, the internal weights will get adjusted. Eachtime, we'll get a better and better match until eventually the model has been trained. 

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-27_at_7.37.09 PM.png" width="80%" />

We can be quite happy with the results of coming in  and out. The important thing to notice here is that the output is generated using only the vector in the middle so that vector contains the meaning of that image and we call that the embedding. 

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-27_at_7.37.27 PM.png" width="80%" />



We'll go and code it in a minute but this is how the model looks on the inside. 
So in here, we can see a group of dense layers and then you can see that as we pass the image through dense layers it gets compressed through 256 and 128 dimensions until we reach the two dimensions and then likewise for the decoder which will take the vector embedding from two dimensions into 128, 256 until we reach the final output. 

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-27_at_8.04.47 PM.png" width="80%" />

The reason we chose the vector embedding in the middle to have two dimensions is purely to make it easier for us to visualize it during the lessons but in fact very often vector embeddings have way more dimensions than just two, often reaching a thousand or more. And here's a nice example of how we can take any kind of data, so we could take an image and then convert it into machine-understandable vector embedding, or we could take a whole piece of text and also generate a vector embedding from it. I cannot stress enough, but basically, the vector embedding captures the meaning of the underlying data. And you can think of vector embeddings as machine-understandable format of the data.

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-27_at_8.05.33 PM.png" width="80%" />

So, let's go and see how it all works in code. 

The first step, we need to load in some libraries, and we'll be using TensorFlow to make it all work. 

```python
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from scipy.stats import norm
```




Very nice! We'll be using MNIST to load our data set. We get a training set and a test set, which we can use later on. We need to normalize this data. And in reality, what we are doing is basically taking this 28 by 28 image and 
then turning into a flattened structure.

```python
# Load data – training and test
(x_tr, y_tr), (x_te, y_te) = mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11490434/11490434 [==============================] - 0s 0us/step



```python
#Normalize and Reshape images (flatten)
x_tr, x_te = x_tr.astype('float32')/255., x_te.astype('float32')/255.
x_tr_flat, x_te_flat = x_tr.reshape(x_tr.shape[0], -1), x_te.reshape(x_te.shape[0], -1)
```

If we print out the shapes of what we had before and after, we'll see that we had 60,000 training objects, which before were 
28 by 28, and now they are 784 dimensions, repeat the same steps for the test data. Lets setparameters for our model. 

```python
print(x_tr.shape, x_te.shape)
print(x_tr_flat.shape, x_te_flat.shape)
```

    (60000, 28, 28) (10000, 28, 28)
    (60000, 784) (10000, 784)

The batch size here is gonna be 100 objects, and it will run across 50 epochs. the idea is that we'll start with 256 dimensions for a hidden state and the objective is to generate vector embeddings of two dimensions. 

```python
# Neural Network Parameters
batch_size, n_epoch = 100, 50
n_hidden, z_dim = 256, 2
```

Now, let's have a look at an example of an input image and you can see that this one looks very much like a zero which is great.

```python
# Example of a training image
plt.imshow(x_tr[1]);
```


Next, we need to construct a 
sampling function which will allow us to grab a number of images 
during the training phase. Next, we need to construct an 
encoder and just like I mentioned earlier on, we'll have 
two dense layers. 


```python
# sampling function
def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
    return mu + K.exp(log_var) * eps
```

This Python code defines a function called `sampling` typically used in the context of variational autoencoders (VAEs), a type of neural network used for generative tasks.

Here's a breakdown of the function:

1. **Function Definition**: `def sampling(args)`: The function `sampling` takes a single argument `args`, which is expected to be a tuple containing two elements: `mu` and `log_var`.

2. **Arguments Unpacking**: `mu, log_var = args`: This line unpacks the `args` tuple into two variables: `mu` and `log_var`. `mu` represents the mean of the latent variable distribution in the VAE, and `log_var` is the logarithm of the variance of this distribution.

3. **Random Noise Generation**: `eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)`: This line generates random noise using a normal distribution. The function `K.random_normal` is likely from a deep learning library (like Keras), denoted by `K`. The noise is generated with a mean of 0 and a standard deviation of 1. The shape of the noise is determined by `batch_size` and `z_dim`, which represent the number of samples and the dimensionality of the latent space, respectively.

4. **Reparameterization Trick**: `return mu + K.exp(log_var) * eps`: This line applies the reparameterization trick. In VAEs, direct sampling from the latent variable distribution is not feasible for backpropagation since it involves a random step. Instead, the reparameterization trick is used: the latent variable is expressed as a deterministic function of the input (here, `mu` and `log_var`) plus some randomness (`eps`). `K.exp(log_var)` converts the logarithm of the variance back to the variance. Multiplying this with `eps` (the random component) and adding `mu` gives a sample from the latent distribution.

In summary, this function is a critical part of a VAE's architecture, allowing it to sample from the latent space in a way that's amenable to gradient-based optimization, thereby enabling the network to learn efficient data representations.


#### Example Data of the function above

- **Batch Size (`batch_size`)**: 10
  - This represents the number of data points we're sampling in each batch.

- **Latent Space Dimension (`z_dim`)**: 5
  - This indicates the dimensionality of the latent space from which we're sampling.

- **Mean (`mu`)**: A zero vector of shape (10, 5)
  - In a typical scenario, `mu` would be learned from the data. For simplicity, let's assume it's a vector of zeros, representing the mean of the latent distribution for each dimension.

- **Log-Variance (`log_var`)**: A zero vector of shape (10, 5)
  - Similar to `mu`, `log_var` is usually learned. Here, we assume it's a vector of zeros, which implies a variance of 1 for each dimension (since exp(0) = 1).

#### Expected Output:

- **Sampled Points**: 10 vectors, each of 5 dimensions
  - These points are sampled from the latent space. Given that both `mu` and `log_var` are vectors of zeros, the sampled points will essentially be random noise scaled by the standard deviation (which is 1 in this case, so it's just standard Gaussian noise).

## Encoder

Next, we need to construct an encoder and just like I mentioned earlier on, we'll have 
two dense layers. The first one will have a dimension of 256. The next one will have dimensions of 128. Next, we need to normalize the data. Now, let's build 
a matching decoder. And in a very similar way, this time starting 
from two dimensions, we'll be moving into 128 and 256 dimensions over 
here. 

```python
# Encoder - from 784->256->128->2
inputs_flat = Input(shape=(x_tr_flat.shape[1:]))
x_flat = Dense(n_hidden, activation='relu')(inputs_flat) # first hidden layer
x_flat = Dense(n_hidden//2, activation='relu')(x_flat)  # second hidden layer
```

```python
# hidden state, which we will pass into the Model to get the Encoder.
mu_flat = Dense(z_dim)(x_flat)
log_var_flat = Dense(z_dim)(x_flat)
z_flat = Lambda(sampling, output_shape=(z_dim,))([mu_flat, log_var_flat])
```

## Decoder Function

```python
#Decoder - from 2->128->256->784
latent_inputs = Input(shape=(z_dim,))
z_decoder1 = Dense(n_hidden//2, activation='relu')
z_decoder2 = Dense(n_hidden, activation='relu')
y_decoder = Dense(x_tr_flat.shape[1], activation='sigmoid')
z_decoded = z_decoder1(latent_inputs)
z_decoded = z_decoder2(z_decoded)
y_decoded = y_decoder(z_decoded)
decoder_flat = Model(latent_inputs, y_decoded, name="decoder_conv")

outputs_flat = decoder_flat(z_flat)
```

And this is our loss function to train the autoencoder, which is also known as variational autoencoder. 

```python
# variational autoencoder (VAE) - to reconstruction input
reconstruction_loss = losses.binary_crossentropy(inputs_flat,
                                                 outputs_flat) * x_tr_flat.shape[1]
kl_loss = 0.5 * K.sum(K.square(mu_flat) + K.exp(log_var_flat) - log_var_flat - 1, axis = -1)
vae_flat_loss = reconstruction_loss + kl_loss

# Build model
#  Ensure that the reconstructed outputs are as close to the inputs
vae_flat = Model(inputs_flat, outputs_flat)
vae_flat.add_loss(vae_flat_loss)
vae_flat.compile(optimizer='adam')

```


Basically the idea here that is optimized to train too much so that the inputs match outputs in a really good fashion. 

## Training

And now, that we have all the pieces, we can begin training that will run across 50 epochs and then each time training across 100 objects.

So, 
this will take a few minutes, so we'll speed it up in post. 

```python
# train
vae_flat.fit(
    x_tr_flat,
    shuffle=True,
    epochs=n_epoch,
    batch_size=batch_size,
    validation_data=(x_te_flat, None),
    verbose=1
)
```
Epoch output is listed below


    Epoch 1/50
    600/600 [==============================] - 8s 12ms/step - loss: 196.4330 - val_loss: 172.3997
    ..
    Epoch 49/50
    600/600 [==============================] - 6s 11ms/step - loss: 134.3089 - val_loss: 136.0062
    Epoch 50/50
    600/600 [==============================] - 6s 11ms/step - loss: 133.8068 - val_loss: 136.0658
    <keras.src.callbacks.History at 0x7f9123babac0>
    Visualize Embeddings
    # Build encoders
    encoder_f = Model(inputs_flat, z_flat)  # flat encoder

### Visualize Embeddings

Now, that the training is done, we can go and visualize our data. 
Let's start and build a flat encoder, and then we can add a piece of code to plot 
our vector embeddings onto a graph.

```python
# Build encoders
encoder_f = Model(inputs_flat, z_flat)  # flat encoder
```


```python
# Plot of the digit classes in the latent space
x_te_latent = encoder_f.predict(x_te_flat, batch_size=batch_size,verbose=0)
plt.figure(figsize=(8, 6))
plt.scatter(x_te_latent[:, 0], x_te_latent[:, 1], c=y_te, alpha=0.75)
plt.title('MNIST 2D Embeddings')
plt.colorbar()
plt.show()
```

<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-28_at_7.11.45 PM.png"  width="80%"/>

And you can see in here that similar vectors are clustered together within the vector embedding space, and then just like you can see that zeros are close together here, nines are close together in here, and the whole space is actually demonstrated or displayed in a two-dimensional space. And those two dimensions are the two dimensions that we have inside the vector embedding. So now, we can get into a phase of comparing vector embeddings.


## Example: compare three embeddings

So, let's give it a new section, and let's grab 
three different images. So, we'll grab one zero, 
which is this one. We can grab another zero. 
We have this one. And let's grab one image that represents 
digit number one. 
```python
plt.imshow(x_te_flat[10].reshape(28,28));
```
<img src="/deeplearningai/vector-databases-embeddings-applications/images/L1-embeddings-image-0.png" width="30%"/>

```python
plt.imshow(x_te_flat[13].reshape(28,28));
```
<img src="/deeplearningai/vector-databases-embeddings-applications/images/L1-embeddings-image-1.png"  width="30%" />



```python
(x_te_flat[2].reshape(28,28));
```

<img src="/deeplearningai/vector-databases-embeddings-applications/images/L1-embeddings-image-2.png"  width="30%"/>

So, if we grab these three 
objects and we can call our function to generate 
vector embeddings. So, zero A, zero B, and one 
will contain the vector embedding values that we need. And 
if we print them, you can see the vectors as follows and you 
can already see that the two zeros are kind of similar to 
each other while the vector that represents digit one is actually quite 
different. We can also do something very similar with text embeddings. 

```python
# calculate vectors for each digit
zero_A = x_te_latent[10]
zero_B = x_te_latent[13]
one = x_te_latent[2]

print(f"Embedding for the first ZERO is  {zero_A}")
print(f"Embedding for the second ZERO is {zero_B}")
print(f"Embedding for the ONE is         {one}")
```
[[ 0.3706197   0.2641425   0.21265654 ...  0.14994532 -0.2579492
  -0.2397075 ]
 [ 0.66933304  0.40094963 -0.48208407 ...  0.10645866 -1.5067165
  -0.01547357]
 [-0.2655591   0.11172403 -0.14733036 ...  0.42197466  0.88394594
   0.10763935]]


<img src="/deeplearningai/vector-databases-embeddings-applications/images/Screenshot_2023-12-28_at_7.20.28 PM.png"  width="60%"/>

### Euclidean Distance(L2)
The length of the shortest path between two points or vectors.

<img src="/deeplearningai/vector-databases-embeddings-applications/images/L1-embeddings-cosinedistance.png"   width="60%"/>

``python
# Euclidean Distance
L2 = [(zero_A[i] - zero_B[i])**2 for i in range(len(zero_A))]
L2 = np.sqrt(np.array(L2).sum())
print(L2)
```


```python
#An alternative way of doing this
np.linalg.norm((zero_A - zero_B), ord=2)
```


```python
#Calculate L2 distances
print("Distance zeroA-zeroB:", np.linalg.norm((zero_A - zero_B), ord=2))
print("Distance zeroA-one:  ", np.linalg.norm((zero_A - one), ord=2))
print("Distance zeroB-one:  ", np.linalg.norm((zero_B - one), ord=2))
```

### Manhattan Distance(L1)
Distance between two points if one was constrained to move only along one axis at a time.

<img src="/deeplearningai/vector-databases-embeddings-applications/images/L1-embeddings-manhattandistance.png"   width="60%"/>

```python
# Manhattan Distance
L1 = [zero_A[i] - zero_B[i] for i in range(len(zero_A))]
L1 = np.abs(L1).sum()

print(L1)
```


```python
#an alternative way of doing this is
np.linalg.norm((zero_A - zero_B), ord=1)
```


```python
#Calculate L1 distances
print("Distance zeroA-zeroB:", np.linalg.norm((zero_A - zero_B), ord=1))
print("Distance zeroA-one:  ", np.linalg.norm((zero_A - one), ord=1))
print("Distance zeroB-one:  ", np.linalg.norm((zero_B - one), ord=1))
```

### Dot Product
Measures the magnitude of the projection of one vector onto the other.

<img src="/deeplearningai/vector-databases-embeddings-applications/images/L1-embeddings-dotproduct.png"   width="60%"/>

```python
# Dot Product
np.dot(zero_A,zero_B)
```


```python
#Calculate Dot products
print("Distance zeroA-zeroB:", np.dot(zero_A, zero_B))
print("Distance zeroA-one:  ", np.dot(zero_A, one))
print("Distance zeroB-one:  ", np.dot(zero_B, one))
```

### Cosine Distance
Measure the difference in directionality between vectors.

<img src="/deeplearningai/vector-databases-embeddings-applications/images/L1-embeddings-cosinedistance.png"   width="60%"/>


```python
# Cosine Distance
cosine = 1 - np.dot(zero_A,zero_B)/(np.linalg.norm(zero_A)*np.linalg.norm(zero_B))
print(f"{cosine:.6f}")
```


```python
zero_A/zero_B
```


```python
# Cosine Distance function
def cosine_distance(vec1,vec2):
  cosine = 1 - (np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
  return cosine
```


```python
#Cosine Distance
print(f"Distance zeroA-zeroB: {cosine_distance(zero_A, zero_B): .6f}")
print(f"Distance zeroA-one:   {cosine_distance(zero_A, one): .6f}")
print(f"Distance zeroB-one:   {cosine_distance(zero_B, one): .6f}")
```

## Now with the sentence embeddings!

Dot Product and Cosine Distance are commonly used in the field of NLP, to evaluate how similar two sentence embeddings are.
So here we will only use those two.

- embedding0 - 'The team enjoyed the hike through the meadow'

- embedding1 - The national park had great views'

- embedding2 - 'Olive oil drizzled over pizza tastes delicious'


```python
#Dot Product
print("Distance 0-1:", np.dot(embedding[0], embedding[1]))
print("Distance 0-2:", np.dot(embedding[0], embedding[2]))
print("Distance 1-2:", np.dot(embedding[1], embedding[2]))
```


```python
#Cosine Distance
print("Distance 0-1: ", cosine_distance(embedding[0], embedding[1]))
print("Distance 0-2: ", cosine_distance(embedding[0], embedding[2]))
print("Distance 1-2: ", cosine_distance(embedding[1], embedding[2]))

```