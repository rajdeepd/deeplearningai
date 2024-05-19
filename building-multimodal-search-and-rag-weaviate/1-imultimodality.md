---
layout: default
title: 1. Multi Modality
nav_order: 2
description: "Introduction to Multimodal Search and RAG with Weaviate"
has_children: false
parent:  Building Multimodal Search and RAG - Weaviate
---

In thislesson you learn about multi-modality, what multi-modal models are and what specifically how to teach a computer the concepts of understanding multi-modal data through the process of contrastive representation learning.
All right. Let's dive in.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-16_at_6.28.18_PM.png"  width="80%" /> 

To stir your imagination,Lets start with the live multi-modal search demo to see how you can search across different types of content.
Here you can see that I can provide a text input so I can search for a bunch of lions and I can get back images, audio and video files.
And in this course we'll go over how this actually works.


We will explain a technology that powers this app. You will learn hands on how to build similar functionality.
We won't necessarily build this app, but by the end of this course,
you should know enough to build apps like this one and more.
So why should you learn about multimodality?
Multimedia content is all around us.
Whether we are searching through our favorite songs, looking for movies
to watch, browsing for things to buy, or looking for a Wikipedia article.
Everything that we do starts with search. But we don't want to just search through text,
we want to set to songs, movie trailers, product images.
We want to search for multi-modal data.
But what is multi-modal data anyway?
Multi-modal data is data that comes from different sources.
It can include text, images, audio and video, and much more.
And each of those modalities often describe similar concepts.



We could have a picture of a lion, a paragraph describing the king of the jungle, a video of running lions, or even the sound of a lion roar.
Each modality comes with a different kind of information.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-15_at_11.25.53_AM.png"  width="80%" /> 

And by combining the information, we gain better understanding of the concepts they represent.
Think of it.
<br/>
<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-15_at_11.26.07_AM.png"  width="80%" /> 

It is more impressive or even scary to see and hear a lion roar than just watching quietly.


<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-15_at_11.27.15_AM.png"  width="80%" /> 


After all, it is only when you see the lion and hearing you understand why he's the king of the jungle.
Another motivation behind why we want to learn from multi-modal data is to think about how humans learn.


<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-15_at_11.27.22_AM.png"  width="80%" /> 

Think of a child in their first year of life before they learn to speak,
a lot of their learning is down to interactions with objects they touch, smell, feel, the texture of, or even taste even if it's soap, but also by watching and listening to everything around them. So this foundational knowledge is built using multi-modal interaction with the world and not by using a language.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-16_at_6.38.21_PM.png"  width="80%" />


Now, if you want to build a smarter and more capable AI, it also needs to learn and think about different modality of information, just like humans do. To get computers to work with multi-modal data, we need to first learn about multi-modal embeddings.

Multi-modal embeddingsallow us to represent multi-modal data on the same vector space, so that a picture of a lion will be placed close to a text describing lions and also a lion roar, or a video of running lions will be there. And we can generate these embeddings from many different sources.

Multi-modal embedding models produce a joint embedding space that understan all of your modalities.

You can understand emails, you can understand images, audio files, and much more.
The key concept here is that this model preserves semantic similarity within and across modalities.
That means if you have objects that are similar, regardless of the modality, their vectors will be close together
like a picture of a lion and a related description.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-16_at_6.29.24_PM.png"  width="80%" /> 


While different concepts like lions and trumpet are far from each other in the multi-modal space. In order to start training multi-modal embedding models, you need to start with a model that understands one modality at a time.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-16_at_10.36.06_PM.png"  width="80%" /> 



This individual models are specialized at understanding text, another separate model might capture representation of images and other models specializing in audio and video.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-16_at_10.36.39_PM.png"  width="80%" /> 


The next task is to unify these models so that regardless of modality, similar concepts should result in close vectors.
So on the left hand side, all of these concepts are similar, and therefore the vectors that they generate on the right hand side are also similar to each other.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-16_at_10.36.53_PM.png"  width="80%" /> 

The task of unifying multiple models into one embedding space is done using a process called Contrastive Representation. Learn. Contrastive Representation
Learning is a general purpose process that can be used to train any embedding model, not just multi-modal embedding models.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-16_at_10.37.11_PM.png"  width="80%" /> 


Specifically here, though, it can also be used to unify multiple models into one multi-modal embedding model with the main idea
where we want to create one unified vector space representation for multiple modalities.
We do it by providing our models with positive and negative examples of similar and different concepts.
Then we train our models to pull closer vectors for similar examples and push further vectors for different concepts.
So let's work to a text example to understand how this works.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-17_at_8.06.01_AM.png"  width="80%" /> 

First we need an anchor point. This can be any data point. For example "he could smell the roses." Then we need a positive example.
An example that is similar to the anchor, like "a field of fragrant flowers." Finally we need a negative example, one that is dissimilar to the anchor, like "the lion roar majestically."

Now, we can get the vector embedding for each data point, and we want to push away the negative vector from the anchor and pull the positive vector closer to the anchor.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-17_at_8.06.30_AM.png"  width="80%" /> 

We can use the same method to train an image model where the anchor could be a picture of a German shepherd.
The positive example could be a grayscale version of the anchor, while the negative example could be a picture of an owl. Now again, the task is to push away the negative example and pull closer to positive example.


<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-17_at_9.20.30_AM.png"  width="80%" /> 

The pushing and pulling process is achieved with the contrastive loss function.
First, we need to encode the anchor and the examples into vector embeddings.
Then we calculate the distance between the anchor and the examples.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-17_at_8.06.01_AM.png"  width="80%" /> 
<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-17_at_8.06.30_AM.png"  width="80%" /> 
<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-17_at_9.20.30_AM.png"  width="80%" /> 

During the training process, we want to minimize the distance between the anchor vector and positive example vectors, while at the same time maximizing the distance to the negative example vectors.

Let us expand the concept of contrastive learning to multi-modal data. We can provide positive and negative examples in different modalities.
Given our anchor is a video of running lions, we can provide contrastive examples as images and text.
Then we can apply pushing and pulling across modalities.
As a result align the model to work in the same vector space across all modalities.

One tricky part can be finding enough of anchors in contrastive examples.in a clip paper from 2021,
they took images as the corresponding captions, each representing a different modality.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-17_at_12.24.36_PM.png"  width="80%" /> 


The picture in its caption represented an anchor in a positive example, as it represented on the diagonal of this matrix,
while any other random pairing of a picture
and a caption is likely to be a negative example.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-17_at_12.25.20_PM.png"  width="80%" /> 


And with that, they were able to apply contrastive loss
function to train the text and image multi-modal model.
Now let's see how a contrastive function looks like.
First you need encoding function that can convert the anchor and contrastive examples into vectors of the same dimension.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-17_at_12.25.27_PM.png"  width="80%" /> 


Here we've got a function $f$ that takes an image and returns a vector $q$.
And here we've got a function $g$ which takes a video and generates vector $k$.
then we take those vector representations and the numerator
you've got a similarity between positive examples.
So this could be the image of a lion and a video of running lions.
You want the similarity to be as high as possible. In the denominator, he got a negative example.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-17_at_12.25.52_PM.png"  width="80%" /> 


Say the image of a lion and a couple of kittens on the bicycle.
This, in fact, is one of many negative examples that you need to sum up.
You want this formula to return a probability, so in the denominator, you normalize by providing the positive example from the numerator again. And up front you have a negative on this loss function,
which means that you actually want to minimize it. And by minimizing this the positive video embeddings will be pulled closer to the anchor image embedding and the negative video will be pushed away from the anchor image.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-05-17_at_12.26.09_PM.png"  width="80%" /> 

You can do this for all the modalities one by one for audio, for text, for video and many more. This is what they did in the image bind paper. Let's now see all of this in practice. In the lab you train an embedding model using contrastive loss. Then visualize the learned vector space.

In this lab, you train a neural network
to learn embeddings for the MNIST image data set.
But let's start by running this code, which was simply ignore any warnings that are not necessarily important for us to analyze.
So first thing you are going to do is import the required libraries.
You're going to use PyTorch for training the model.
And a set of supporting libraries. The most important ones are libraries for visualization Like Plotly and umap.
Last thing that you need to import is the MNIST dataset class, which you used to get positive and negative examples to train and test on.




* In this classroom, the libraries have been already installed for you.
* If you would like to run this code on your own machine, you need to install the following:
```
    !pip install -q accelerate torch
    !pip install -U scikit-learn
    !pip install umap-learn
    !pip install tqdm
```


```python
import warnings
warnings.filterwarnings('ignore')
```
The MNIST dataset is based on images of digits from 0 to 9 , and each image of a digit is labeled with the value that it corresponds to, so that we know whether it represents zero or a five or a nine, and the dataset class provides you with an anchor, which is a digit, and a positive and negative example, which are also digit examples.
Just like we talk about in the slides.
For example, if the anchor is a five, then a positive example would be another image for a five.
While a negative example would be an image of let's say six or a seven.
And if you're curious about the MNIST dataset
code, feel free to review the MNIST dataset.py file.
And in here there are two key parts that you should pay attention to.
This piece of code over here takes care of labeling positive and negative examples, so that when the label from the anchor matches the I, that automatically means that this is a positive example.

This is when you have a seven and a seven.
While if these values differ so maybe they increased seven.
While the example is five then this is assigned to the negative labels list.
And the second key part is the code over here, which takes
care of allocating the ideal distance metric that you will use during training.
And since you will use cosine similarity, the ideal similarity
4 between positive examples and the anchor is set to one.
While for negative examples this is set to zero.
And in a nutshell, these are the two key parts that allow you to find the positive
and negative examples and allocate ideal distances to each example.


All right. You're doing great.
Now, let's define a neural network architecture
that would take the MNIST images and output them, 64 dimensional vectors.
So this is a simple architecture we have two convolutional layers.
And the point of these
convolutional layers is to extract the visual features of a digit.
Then you have two feedforward linear layers here.
And their job is to make visual features learned by the convolution layer and learn how to turn them into 64 dimensional representations.

## Import libraries


```python
# Import neural network training libraries
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Import basic computation libraries along with data visualization and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.decomposition import PCA
import umap
import umap.plot
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = 'iframe'

# Import our data class which will organize MNIST and provide anchor, positive and negative samples.
from mnist_dataset import MNISTDataset
```

## Load MNIST Dataset


Let's get back to the training notebook.
So now let's load the data from the MNIST data set.
And here, the training data set and the validation data set that, you need to load.
And then let's execute this, and this will load the data set for you.
Okay, great.
And once the data set is loaded, you need to set up PyTorch data loaders.
That will feed the neural network to train on.
So feel free to modify this batch size over here.
if you want to see how that changes the convergence of the training.



```python
# Load data from csv
data = pd.read_csv('digit-recognizer/train.csv')
val_count = 1000
# common transformation for both val and train
default_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

# Split data into val and train
dataset = MNISTDataset(data.iloc[:-val_count], default_transform)
val_dataset = MNISTDataset(data.iloc[-val_count:], default_transform)
```
## Setup our DataLoaders


```python
# Create torch dataloaders
trainLoader = DataLoader(
    dataset,
    batch_size=16, # feel free to modify this value
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    prefetch_factor=100
)

valLoader = DataLoader(val_dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    prefetch_factor=100
)
```

Let us use the above data loaders to visualize the anchor points
and the corresponding positive and negative examples.

### Visualize datapoints

First add a helper function that will display provided images.
And then you need to loop through
the batch of the data sets from the train data loader.
And from here would you be grabbing the anchor images, the contrastive images their ideal distances and the labels.



```python
# Function to display images with labels
def show_images(images, title=''):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(9, 3))
    for i in range(num_images):
        img = np.squeeze(images[i])
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    fig.suptitle(title)
    plt.show()

# Visualize some examples
for batch_idx, (anchor_images, contrastive_images, distances, labels) in enumerate(trainLoader):
    # Convert tensors to numpy arrays
    anchor_images = anchor_images.numpy()
    contrastive_images = contrastive_images.numpy()
    labels = labels.numpy()
    
    # Display some samples from the batch
    show_images(anchor_images[:4], title='Anchor Image')
    show_images(contrastive_images[:4], title='+/- Example')
    
    # Break after displaying one batch for demonstration
    break

```

And if you run this, you should get examples of, like, four images of each.
And here you can see one anchor,
which is a nine, and a positive example, which is another nine.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/1-image1.png" />

There is an eight and a six, this is a negative example, another positive example.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/1-image2.png" />


Four and a seven, which should be used as a negative example.
That is basically what we will use for pushing and pulling to train this, neural network.




## Build Neural Network Architecture


Now, let's define a neural network architecture that would take the MNIST images and output them, 64 dimensional vectors.
This simple architecture we have two convolutional layers. The point of these convolutional layers is to extract the visual features of a digit. You have two feedforward linear layers here, their job is to make visual features learned by the convolution layer and learn how to turn them into 64 dimensional representations.
All of this is combined in the forward function.

In summary, you pass an image at the top, then you run through the two convolutional layers,
flatten, the output, then finally pass it through a linear function.
At the end you get a 64 dimensional vector. That's basically how your neural network architecture looks like.

```python
# Define a neural network architecture with two convolution layers and two fully connected layers
# Input to the network is an MNIST image and Output is a 64 dimensional representation. 
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Dropout(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Dropout(0.3)
        )
        self.linear1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
        )

    def forward(self, x):
        x = self.conv1(x) # x: d * 32 * 12 * 12
        x = self.conv2(x) # x: d * 64 * 4  * 4 
        x = x.view(x.size(0), -1) # x: d * (64*4*4)
        x = self.linear1(x) # x: d * 64
        return x
```

And that is basically how your neural network architecture looks like.
Now, to train the neural network, you use the contrastive loss function, you would take these 64 dimensional vector representations for both the anchor and then either a positive or negative example. Then you make sure that the ideal distances are met.

## Contrastive Loss Function

Like discussed earlier, the ideal cosine distance for a positive sample is one and a negative is zero.
This is probably the most important part of this. Let us dive into it.
We will the contrastive loss as calculating a cosine similarity between two points.
The forward function you have the *anchor* and the *contrastive example* which could be either a positive or a negative one and together with their ideal distance. The forward function is implemented as two step thing : first, we calculate the score.
The cosine similarity between the anchor and the contrastive images. 

```python
# The ideal distance metric for a positive sample is set to 1, for a negative sample it is set to 0      
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.similarity = nn.CosineSimilarity(dim=-1, eps=1e-7)

    def forward(self, anchor, contrastive, distance):
        # use cosine similarity from torch to get score
        score = self.similarity(anchor, contrastive)
        # after cosine apply MSE between distance and score
        return nn.MSELoss()(score, distance) #Ensures that the calculated score is close to the ideal distance (1 or 0)


Then in the second step, this score is, compared to the ideal distance that, you expect. 

### Key Information

Lower distance between these two scores mean lower loss. You want to eventually minimize this loss. By the time this neural network is trained, the calculated score should be very close to the expected ideal distance.

#### Define the Training Configuration

In this step you use CPU for training as the default option. However, if the GPU or Cuda are available, then in this case that would be used instead of the CPU.


```python
net = Network()

device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    
net = net.to(device)
device
```

You also need to add in the configuration parameters for the neural network training.

```python
# Define the training configuration
optimizer = optim.Adam(net.parameters(), lr=0.005)
loss_function = ContrastiveLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)
```

We are using Adam optimizer with ContrastiveLoss function defined above.
This step will do the gradient descent. Then you need to set up contrastive loss function.
And here is your training scheduler. Now, you are getting to the fun part.


The code that executes the training of this model.
To help us set it all up, we need, a checkpoints folder, where the results of the training will be saved into, that you could actually use later on to reload all the training back.
Let us have a look at the training loop.You can set it to run over any number of epochs.


## Training Loop

The code that executes the training of this model.
First to help us set it all up, we need, a checkpoints folder, where the results of the training will be saved into, 3 that you could actually use later on to reload all the training back.


```python
import os

# Define a directory to save the checkpoints
checkpoint_dir = 'checkpoints/'

# Ensure the directory exists
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
```

### Model Training

Let us have a look at the training loop.
You can set it to run over any number of epochs.
The training loop goes through and trains the model, which goes through 3 forward propagation and backwards propagation until it converges.
One of the key parts to see in here is, the loss function,
which takes the anchor and the contrastive vectors, together with their expected
distances and calculates the loss, for each data point.
The loss is then added up, which at the end of the epoch is used to calculate the average loss for each epoch, which is how you know,
whether the model training is improving or not.

As each epoch is completed, the loop saves the results into the checkpoint folder.
And finally, what the function returns back is the trained model together with, an array of the losses, epoch by epoch.



```python
def train_model(epoch_count=10):#
    net = Network()
    lrs = []
    losses = []

    for epoch in range(epoch_count):
        epoch_loss = 0
        batches=0
        print('epoch -', epoch)
        lrs.append(optimizer.param_groups[0]['lr'])
        print('learning rate', lrs[-1])
    
        for anchor, contrastive, distance, label in tqdm(trainLoader):
            batches += 1
            optimizer.zero_grad()
            anchor_out = net(anchor.to(device))
            contrastive_out = net(contrastive.to(device))
            distance = distance.to(torch.float32).to(device)
            loss = loss_function(anchor_out, contrastive_out, distance)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        losses.append(epoch_loss.cpu().detach().numpy() / batches)
        scheduler.step()
        print('epoch_loss', losses[-1])
    
        # Save a checkpoint of the model
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt')
        torch.save(net.state_dict(), checkpoint_path)

    return {
        "net": net,
        "losses": losses
    }
```
This is in a nutshell, how the training works.
Please note, though, that the training process is rather slow, so it can take, 2 to 3 minutes per epoch.
It can take quite a while to run across, ten, twenty or even hunder of them.

The model presented above is ready for training; however, please be aware that it may take several minutes to train. As a backup plan, we provide the option to load a pre-trained model.

### Load from Backup

But don't worry. We have a backup plan, and we already pre-trained this model over 100 epochs.
And in the interest of time,
I suggest loading the pre-trained model from the provided checkpoint, instead of training the model from scratch.

```python
def load_model_from_checkpoint():
    checkpoint = torch.load('checkpoints/model_epoch_99.pt')
    
    net = Network()
    net.load_state_dict(checkpoint)
    net.eval()

    return net
```

Now, run the following code to get your model.
And by default, this will try and load the model from the checkpoint.
However, if you would like to run the full training
yourself, you can change this train flag to true.
remember, this is a lengthy process, so you need to be very patient
when you do run this. And let's run this and you should straight away
get ready model.


Set the `train` variable to `TRUE` if you'd like to train the model, otherwise you will load a trained checkpoint of the model.

### Get the Model

<p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> <b>(Note: <code>train = False</code>):</b> We've saved the trained model and are loading it here for speedier results, allowing you to observe the outcomes faster. Once you've done an initial run, you may set <code>train</code> to <code>True</code> to train the model yourself. This can take some time to finsih, depending the value you set for the <code>epoch_count</code>.</p>


```python
train = False # set to True to run train the model

if train:
    training_result = train_model()
    model = training_result["net"]
else:
    model = load_model_from_checkpoint()
```

### Visualize the loss curve for your trained model

If you did choose to train the model yourself,
then you can plot out how the loss was, changing over the epochs.
And you can see that in our case, the training was pretty much settled around the 20 epochs.
So most of the learning was done in the first 5 to 10 ,
and then around 20 were already settled

```python
from IPython.display import Image

if train:
    # show loss curve from your training.
    plt.plot(training_result["losses"])
    plt.show()
else:
    # If you are loading a checkpoint instead of training the model (train = False),
    # the following line will show a pre-saved loss curve from the checkpoint data.
    display(Image(filename="images/loss-curve.png", height=600, width=600))
```
