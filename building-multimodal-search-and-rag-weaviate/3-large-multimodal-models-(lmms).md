---
layout: default
title: 3. Large Multimodal Models (LMMs)
nav_order: 4
description: "Multimodal Search"
has_children: false
parent:  Building Multimodal Search and RAG - Weaviate
---

In this lesson you learn how large language models work and how they understand text.
Then, you will learn how to combine LLMs and multimodal modals into language vision models using a process called Visual Instruction Tuning?

Screenshot_2024-06-07_at_10.39.54_PM.png



And finally, you use all these models in practice.
All right, let's go.
Current LLMs are all generative Pre-trained transformers.
For example, Llama2 , Chat-GPT or Mistral.
This class of models are autoregressive because they generate one token or one word piece at a time.
Further tokens that are generated only depend on previously provided or generated tokens.
These models have been trained in unsupervised manner by predicting the next word on trillions of tokens.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-07_at_10.40.08_PM.png"  width="80%" /> 


In this training process, the output a probability over all possible next tokens.
And we train them to get this correct.
Let's take a look at an example.
Jack and Jill
went up the blank, which outputs a score for every token.
We're more pro tokens like mountain and hill will have higher scores, while less protocol tokens like apple and llama will have lower scores.
You can think of these scores as a normalized probabilities.
This scores are then normalized two percentage points like this.
Given a prompt: "the Rock".
We want to see how the model completes this.
We use one or two vectors to represent each word and then look up the embedding for each token.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-07_at_10.40.25_PM.png"  width="80%" /> 


Once we get the embedding for a token, the transformer model would try to generate the next word by paying attention to the first word.
It always starts off with beginning of the sentence token, and because we know the first two words are the rock, we can force the model to output them.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-07_at_10.40.55_PM.png"  width="80%" /> 


And in fact, this is actually how the model is trained.
And once the model outputs rock,
we input this word representation as one on the vector back in.
Now the model looks up the embedding for rock
and also pays attention to previous words in the same sentence and outputs
a probability over the next possible tokens that we can sample from.
And here we can see that we sampled the word rolls.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-07_at_10.45.05_PM.png"  width="80%" /> 

We pass this forward and generate the next word, which is along.
This can keep going until we hit the token limit or the end of the sentence token.


Let's say we get the word skips and we pass this word forward, and the next word that we get is fast.
If we keep something, we can get a completely different generated response.

### Image Classification

Here's a simple example of an image classification model.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-16_at_4.06.24_PM.png"  width="80%" /> 

It takes in an image and outputs a class label. 


<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-16_at_4.06.39_PM.png"  width="80%" /> 

A vision transformer works quite well for this classification task. It processes an image as patterns instead of individual pixels, which makes the analysis much more efficient.
So let's look at this in more detail.

Each part of the image gets vectorized and passed into the transformer model, he transformer, can choose to pay attention to any part and is optimized to output the correct label.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-16_at_4.07.31_PM.png"  width="80%" /> 


## Visual Instruction tuning

Now let's see how you can use visual instruction tuning to train and LLMto process images along with text, given an image and a text instruction. And in this case, it is the starry night.


<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-16_at_7.58.15_PM.png"  width="80%" /> 

And the question is "who drew this painting?"
You can train the model to output the correct answer in text, which is, of course, Vincent Van Gogh.
Vow let's look at an example of visual instruction tuning. You are going to start off with your image, the Starry night painting, and then you're going to cut it up into patches. And you also have a text instruction "who do this painting." so you are going to take the ourchase of our image and embed them into vectors as seen here. 

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-16_at_7.58.27_PM.png"  width="80%" /> 

You are going to take your tokens from our sentence into an instruction and you're going to embed that into vectors as well. Now, the language model is going to be trained to understand and pay attention to both of the image patch tokens well as the language token, and it has to output the correct tokens or the answer. Vincent Van Gogh.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-16_at_7.58.42_PM.png"  width="80%" /> 

This is known as visual instruction tuning because you are given a visual as well as instruction and you know what the right answer is.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/Screenshot_2024-06-16_at_7.59.24_PM.png"  width="80%" /> 

You can optimize the probability that the model generates the right outputs token in the process, learns to understand images.

You can optimize the probability that the model generates the right outputs token in the process, learns to understand images.
After an LLM is trained using the visual instruction tuning,
you can now process images as well as text. You can think of the model as a large multimodal model, an LLM. You can also ask questions about objects in the images for example, ask for detailed structure description of the image content like this.
'Describe this picture for me." Let's now see all of this in practice.

### Lab

In this lab you use images and text as input,then you get LLMs to reason over it.

* In this classroom, the libraries have been already installed for you.
* If you would like to run this code on your own machine, you need to install the following:
* 
```
    !pip install google-generativeai

```
Lets start by adding a command that will ignore all the unnecessary warnings.
Let's do a bit of a setup.
In this lesson will be using Gemini provision. What we need to do is load our API keys and then also as part of this, what we have is a genai library where we need to pass in the key.
Let's load some helper functions. Now what we need is a function that will take a piece of text and extract it and turn into readable markdown. Let's construct a function that allows us to call LLMs. That function will take an image path and a prompt.



Note: don't forget to set up your `GOOGLE_API_KEY` to use the Gemini Vision model in the `env` file.

```
   %env GOOGLE_API_KEY=************
```

Check the [documentation](https://ai.google.dev/gemini-api/docs/api-key) for more infomation.


```python
import warnings
warnings.filterwarnings('ignore')
```

## Setup

### Load environment variables and API keys


```python
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file
GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
```


```python
# Set the genai library
import google.generativeai as genai
from google.api_core.client_options import ClientOptions

genai.configure(
        api_key=GOOGLE_API_KEY,
        transport="rest",
        client_options=ClientOptions(
            api_endpoint=os.getenv("GOOGLE_API_BASE"),
        ),
)
```

> Note: learn more about [GOOGLE_API_KEY](https://ai.google.dev/) to run it locally.

## Helper functions


```python
import textwrap
import PIL.Image
from IPython.display import Markdown, Image

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

```

* Function to call LMM (Large Multimodal Model).


```python
def call_LMM(image_path: str, prompt: str) -> str:
    # Load the image
    img = PIL.Image.open(image_path)

    # Call generative model
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()

    return to_markdown(response.text)  
```

## Analyze images with an LMM

First thing we need to do is load that image, next thing that we need to do is call the generative model Gemini provisioned with the prompt and the loaded image.
Finally, we need to return the result. That's where we're going to use that to mark down function, which would take this text from the response and then pass into something nice and readable.

We can start analyzing some images, and we are going to start with this beautiful index historical chart.

```python
# Pass in an image and see if the LMM can answer questions about it
Image(url= "SP-500-Index-Historical-Chart.jpg")
```


```python
# Use the LMM function
call_LMM("SP-500-Index-Historical-Chart.jpg", 
    "Explain what you see in this image.")
```

Let's call our and then function. Given the file and a prompt which says, "explain what you see in this image." I would try to get them to analyze this chart, usually that takes a couple of seconds.
Now we see a nice description which basically says that the image shows historical charts from S\&amp;P 500
and gives us a pretty nice analysis of what we really see in here.
This could be quite helpful,
Lets let's try to analyze something harder. We're going to use the graphic that we use in our slides and we ask the LLM to give us a hand in explaining what this figure is actually used for.
This is the result. 



<p style="background-color:#F8F9F9; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px">
The image shows the historical chart of the S&P 500 index. The S&P 500 is a stock market index that tracks the 500 largest publicly traded companies in the United States. The index is considered to be a leading indicator of the overall U.S. stock market.
The chart shows that the S&P 500 index has been on an overall upward trend since its inception in 1957. However, the index has experienced several periods of volatility, including the 1987 crash and the 2003 lows.
Despite these periods of volatility, the S&P 500 index has continued to climb over the long term. This is because the U.S. economy has continued to grow over time, and companies have generally been able to increase their earnings.
The S&P 500 index is a popular investment for investors who are looking for long-term growth. The index provides exposure to a wide range of companies, and it has historically outperformed other investments, such as bonds and cash.
However, it is important to remember that the S&P 500 index is not without risk. The index can experience periods of volatility, and there is always the possibility that the index could decline in value.
Investors who are considering investing in the S&P 500 index should be aware of the risks involved and should consult with a financial advisor before making any investment decisions.
</p>

## Analyze a harder image

* Try something harder: Here's a figure we explained previously!


```python
Image(url= "clip.png")
```
<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/3-clip.webp" width="80%" />

```python
call_LMM("clip.png", 
    "Explain what this figure is and where is this used.")
```

In here you can see that the model recognize that this was an image that was used for contrastive Pre-training framework, which is actually pretty accurate. Then it explains about like different typesof encoder for text and image and so on and so on. Maybe I should have used it for preparing this lesson.
Here is a fun example that I want to go over. If you look at this, this is just a green blob.
There's nothing special about it. I'm kind of curious what this LLM can come up with when he's trying to analyze this.
Lets ask the LLM to see if it can see something special about this image.

<p style="background-color:#F8F9F9; padding:15px; border-width:3px; border-color:#E5E7E9; border-style:solid; border-radius:6px">
This figure shows a contrastive pre-training framework for learning image-text representations.
The framework consists of two encoders: a text encoder and an image encoder.
The text encoder takes as input a text sequence and outputs a text embedding.
The image encoder takes as input an image and outputs an image embedding.
The text embedding and the image embedding are then used to compute a contrastive loss.
The contrastive loss is a function that measures the similarity between two embeddings.
The goal of the contrastive loss is to pull similar embeddings closer together and push dissimilar embeddings further apart.
By minimizing the contrastive loss, the encoders learn to produce embeddings that are discriminative and useful for downstream tasks such as image-text retrieval and image captioning.
This figure is used in the paper "Unsupervised Learning of Visual Representations by Contrastive Learning" by Armand Joulin, Laurens van der Maaten, Allan Jabri, and Nicolas Vasilache. The paper was published in the proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) in 2017.
</p>


We can see that anyone recognized that there is a hidden message that says you can vectorize the whole world with Weaviate.

## Decode the hidden message

<p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#E5E7E9; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access Utils File and Helper Functions:</b> To access the files for this notebook, 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>


Let's try to run this function that will show us where this message was hidden. And then, in here what we are actually doing is we're looking for anything hat was in the first channel, what a value was over 120. 

```python
Image(url= "blankimage3.png")
```
<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/3-blankimage3.webp" width="40%" />

```python
# Ask to find the hidden message
call_LMM("blankimage3.png", 
    "Read what you see on this image.")
```

And by running this, we can see that this is the message that was really hidden there.

<img src="/deeplearningai/building-multimodal-search-and-rag-weaviate/images/3-blankimage3_decoded.png" width="40%" />

Anything that was over 20 became white and under is black, and that's how the LLM was able to decode the message and tell us about it.

## How the model sees the picture!

LLMs actually don't see the way we see. They can actually see a lot more and be more inquisitive about some of the images that they're looking to and that's how they can decode this kind of stuff. I will include a function in a resources for this notebook.

```python
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

image = imageio.imread("blankimage3.png")

# Convert the image to a NumPy array
image_array = np.array(image)

plt.imshow(np.where(image_array[:,:,0]>120, 0,1), cmap='gray');
```

### Try it yourself!

**EXTRA!**  You can use the function below to create your own hidden message, into an image:


```python
# Create a hidden text in an image
def create_image_with_text(text, font_size=20, font_family='sans-serif', text_color='#73D955', background_color='#7ED957'):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(background_color)
    ax.text(0.5, 0.5, text, fontsize=font_size, ha='center', va='center', color=text_color, fontfamily=font_family)
    ax.axis('off')
    plt.tight_layout()
    return fig
```


If you want to create an image with a hidden message like this one, you'll be able to do it just like that and send it to your friends.

So in this lesson, you learn how to use image vision models and how to actually analyze images to gather with text prompts. And then in the next lesson, you will build a multi model RAG up, See in an accessory.