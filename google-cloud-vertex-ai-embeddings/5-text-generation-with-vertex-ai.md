---
layout: default
title: 5. Text Generation with Vertex AI
nav_order: 6
description: ".."
has_children: false
parent:  Google Cloud Vertex AI Embeddings
---
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

<script type="text/javascript"
  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

Let's consider <math><mi>a</mi><mo>≠</mo><mn>0</mn></math>.

In this course, our objective is to develop a question-answering system. While embeddings alone offer a foundational capability for such a system, integrating the text generation features of advanced language models enables us to enhance its performance significantly. Our journey begins with the necessary setup, including authentication and initialization of the Vertex AI service, after specifying our operational region and initializing the Vertex AI SDK.


```python
from utils import authenticate
credentials, PROJECT_ID = authenticate()
```


```python
REGION = 'us-central1'
```


```python
import vertexai
vertexai.init(project=PROJECT_ID, 
              location=REGION, 
              credentials = credentials)
```

Our focus shifts to employing a text generation model from Vertex AI, moving beyond the embeddings model we previously utilized. We introduce a model known as `text-bison`, distinctively optimized for tasks including sentiment analysis, classification, summarization, and information extraction. It's important to note that `text-bison` excels in scenarios requiring a singular API response, making it less suitable for ongoing dialogues. For applications demanding interactive exchanges (multi-turn dialogue), a different model, `chat-bison`, is recommended.


```python
from vertexai.language_models import TextGenerationModel
```


```python
generation_model = TextGenerationModel.from_pretrained(
    "text-bison@001")
```


Text generation with large language models involves providing textual input, or a prompt, to which the model responds with appropriate output. For our exploratory task, we craft a prompt aimed at generative brainstorming: "I'm a high school student. Recommend me a programming activity to improve my skills." This prompt is then submitted to our model to generate a response.

Upon executing the model with our prompt, we receive valuable suggestions, such as engaging in a programming project that aligns with personal interests or enrolling in programming courses to further develop coding skills. This demonstrates the model's capacity to provide actionable and relevant advice based on the input provided.

```python
prompt = "I'm a high school student. \
Recommend me a programming activity to improve my skills."
```


```python
print(generation_model.predict(prompt=prompt).text)
```

```
* **Write a program to solve a problem you're interested in.** This could be anything from a game to a tool to help you with your studies. The important thing is that you're interested in the problem and that you're motivated to solve it.
* **Take a programming course.** There are many online and offline courses available, so you can find one that fits your schedule and learning style.
* **Join a programming community.** There are many online and offline communities where you can connect with other programmers and learn from each other.
* **Read programming books and articles.** There is a

```


This time, we refine our inquiry by presenting a high school student with a choice between three programming activities: A) learning Python, B) learning JavaScript, or C) learning Fortran, asking which one is recommended and why. This approach narrows down the options, prompting a more focused response from our text generation model. After submitting this specific query and employing the predict function of our model, we observe the model advocates for learning Python. 
```python
prompt = """I'm a high school student. \
Which of these activities do you suggest and why:
a) learn Python
b) learn Javascript
c) learn Fortran
"""
```

The model's suggestion, while informed by the constraints of our query, showcases the importance of precisely framing prompts—a concept known as prompt engineering.



```
I would suggest learning Python. Python is a general-purpose programming language that is easy to learn and has a wide range of applications. It is used in a variety of fields, including web development, data science, and machine learning. Python is also a popular language for beginners, as it has a large community of support and resources available.
```

Prompt engineering involves the strategic crafting of input text to elicit the most useful response from a model, balancing between giving enough context for accurate responses and not overly constraining the possibilities. This technique plays a crucial role in maximizing the effectiveness of large language models for various tasks, including information extraction, sentiment analysis, and more.

Engaging with large language models opens avenues for extracting insights and generating responses that align closely with the input prompt's intentions. For those intrigued by the nuances of prompt engineering and wishing to delve deeper into optimizing interactions with language models, exploring additional resources or courses on the subject is highly recommended. As we proceed, we'll explore another application of these versatile models to demonstrate their capacity for information extraction and other complex tasks

#### Extract information and format it as a table

Essentially, we are transforming information from one format into another. In this case, we have a detailed synopsis for a fictitious movie featuring a wildlife biologist, complete with character names, their professions, and the actors portraying them. Our objective is to prompt the model to identify and extract these specific details. By feeding this extensive narrative to the model, we set the stage for a focused extraction task.

The model successfully identifies the characters, their respective roles, and the actors from the synopsis. To further refine our output, we can instruct the model to reorganize this extracted information into a structured format, such as a table, enhancing readability and organization.

Upon requesting a tabulated representation, the model outputs markdown code, which, when rendered, forms a well-organized table displaying the extracted details in a clear and structured manner. This illustrates the capability of large language models not only to extract vital information but also to reformat it, bridging the gap between raw text and structured data.

In breaking down the process, we started by selecting and importing a text generation model, specifically the TextBison model. After defining our detailed prompt, we executed the predict function with the prompt as its argument. The discussion on input text and model outputs emphasizes that, beyond merely generating subsequent text, these models analyze input to project an array of potential continuations, grounded in the likelihood of various tokens. This refinement in understanding underscores the model's intricate mechanism of generating contextually relevant and structured outputs from unstructured inputs.

```python
prompt = """ A bright and promising wildlife biologist \
named Jesse Plank (Amara Patel) is determined to make her \
mark on the world. 
Jesse moves to Texas for what she believes is her dream job, 
only to discover a dark secret that will make \
her question everything. 
In the new lab she quickly befriends the outgoing \
lab tech named Maya Jones (Chloe Nguyen), 
and the lab director Sam Porter (Fredrik Johansson). 
Together the trio work long hours on their research \
in a hope to change the world for good. 
Along the way they meet the comical \
Brenna Ode (Eleanor Garcia) who is a marketing lead \
at the research institute, 
and marine biologist Siri Teller (Freya Johansson).

Extract the characters, their jobs \
and the actors who played them from the above message as a table
"""
```


```python
response = generation_model.predict(prompt=prompt)

print(response.text)
```

- You can copy-paste the text into a markdown cell to see if it displays a table.

| Character | Job | Actor |
|---|---|---|
| Jesse Plank | Wildlife Biologist | Amara Patel |
| Maya Jones | Lab Tech | Chloe Nguyen |
| Sam Porter | Lab Director | Fredrik Johansson |
| Brenna Ode | Marketing Lead | Eleanor Garcia |
| Siri Teller | Marine Biologist | Freya Johansson |


### Adjusting Creativity/Randomness

We previously introduced the concept of tokens, which are the fundamental units of text processed by a large language model. Tokens can vary from words to subwords or other textual fragments, depending on the tokenization approach employed. While I use the term "tokens," for clarity in this explanation, I'll refer to them as individual words. It's important to remember that in practice, models operate on a more granular level, predicting an array of probabilities for the subsequent tokens.

<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-16_at_5.50.34%E2%80%AFPM.png" width="80%"/>


The process of selecting the next token from this probability array is governed by a decoding strategy. A straightforward method, known as greedy decoding, involves choosing the token with the highest probability at each step. However, this approach might lead to repetitive or mundane outputs.
<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-16_at_6.42.05%E2%80%AFPM.png" width="80%"/>
<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-16_at_6.42.23%E2%80%AFPM.png" width="80%"/>

Alternatively, random sampling from the probability distribution introduces variety but risks generating odd or incoherent responses. Adjusting the level of randomness, a mechanism controlled by a parameter known as temperature, influences the creativity or conventionality of the responses. Lower temperature settings suit tasks requiring precise or narrow responses, such as classification or specific information extraction. In contrast, higher temperatures foster creativity, beneficial for brainstorming or summarization tasks where novel or diverse outputs are desired.

<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-16_at_6.43.01%E2%80%AFPM.png" width="80%"/>
<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-16_at_6.44.45%E2%80%AFPM.png" width="80%"/>
<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-16_at_6.45.57%E2%80%AFPM.png" width="80%"/>


The transformation from raw model outputs, or logits, to a probability distribution involves the softmax function. This distribution reflects the likelihood of each token being the next in sequence. Temperature adjustment essentially modifies this probability distribution, shaping the balance between predictability and novelty in the model's responses.

Applying temperature to the softmax function involves adjusting each logit value by dividing it by the temperature parameter, <math><mi>&#x3B8;</mi></math>. This alteration effectively modulates the distribution of probabilities generated by the softmax function, allowing for controlled variability in the model's output. To visualize, imagine the standard softmax function, and then picture it adjusted by temperature, where each logit, $z$, is divided by <math><mi>&#x3B8;</mi></math>, enhancing our control over the model's predictive behavior.

<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-16_at_6.46.30%E2%80%AFPM.png" width="80%"/>

<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-16_at_6.46.42%E2%80%AFPM.png" width="80%"/>

<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-16_at_6.47.23%E2%80%AFPM.png" width="80%"/>


If the concept of adjusting logits with temperature seems complex, focus instead on grasping the practical impact of temperature on model outputs. Lowering the temperature tightens the probability distribution around the most probable token, making it more likely to be chosen. If we reduce the temperature to zero, the model's selection becomes entirely deterministic, consistently picking the highest probability token.

<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-16_at_6.47.37%E2%80%AFPM.png" width="80%"/>
<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-16_at_6.47.46%E2%80%AFPM.png" width="80%"/>


Conversely, increasing the temperature broadens the probability distribution, giving lower probability tokens a better chance of being selected, thus introducing more diversity into the responses. Within the context of Vertex AI, setting a temperature value within the range of zero to one is recommended, with a starting point of 0.02 often being effective for a variety of tasks, adjustable based on the desired outcome.

To demonstrate, let's experiment with a temperature setting of zero, ensuring a deterministic output where the model will invariably select the most probable token at each step. Consider the prompt: "Complete the sentence. As I prepared the picture frame, I reached into my toolkit to fetch my..." The model's response, influenced by a zero temperature setting, will reflect the most likely continuation based on its training.



So, we will call the predict function as we've done before on our generation model. And we'll pass in the prompt, but this time, we're also going to pass in the temperature value.  



```python
temperature = 0.0
```


```python
prompt = "Complete the sentence: \
As I prepared the picture frame, \
I reached into my toolkit to fetch my:"
```


```python
response = generation_model.predict(
    prompt=prompt,
    temperature=temperature,
)
```


```python
print(f"[temperature = {temperature}]")
print(response.text)
```
And then we can print out this response. So, the model says, as I prepared the picture frame, I reached into my toolkit to fetch my hammer. And that seems 3 like a pretty reasonable response, probably the most likely thing someone would fetch from 3 their toolkit for this particular example. And remember, temperature of 0 is deterministic. So, even if we run this again, we will get the exact same answer. So, let's try this time setting the temperature to 1.

```python
temperature = 1.0
```


```python
response = generation_model.predict(
    prompt=prompt,
    temperature=temperature,
)
```


```python
print(f"[temperature = {temperature}]")
print(response.text)
```

And again, we can call the predict function on our model. And we will print out the result with this different temperature value. And this time, we reached into the toolkit to fetch my saw. I ran this earlier. I saw sandpaper, which 3 I thought was a pretty interesting response.

Summarizing the points discussed 
- You can control the behavior of the language model's decoding strategy by adjusting the temperature, top-k, and top-n parameters.
- For tasks for which you want the model to consistently output the same result for the same input, (such as classification or information extraction), set temperature to zero.
- For tasks where you desire more creativity, such as brainstorming, summarization, choose a higher temperature (up to 1).

The model also actually produced some 3 additional information here as well. So, you can try this out, and you'll get a different response if you run this again. So, I encourage you to try out some different temperature values and see how that changes the responses from the model. 
#### Top P

Now, in addition to temperature, there are two other hyperparameters that you can set to impact the randomness and the output of the model. So, let's return to our example from earlier where we had an input sentence, the garden was full of beautiful, and this probability array over tokens. One strategy for selecting the next token is called TopK, where you sample from a shortlist of the TopK tokens. So, in this case, if we set K to two, that's the two most probable tokens, flowers and trees.

<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-16_at_7.42.23%E2%80%AFPM.png" width="80%"/>
<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-16_at_7.42.37%E2%80%AFPM.png" width="80%"/>


Now, TopK can work fairly well for examples where you have several words that are all fairly likely, but it can produce some interesting or sometimes not particularly great results when you have a probability distribution that's very skewed. So, in other words, you have a one word that's very likely and a bunch of other words that are not very likely. And that's because the top <math><mi>K</mi></math> value is hard coded for a number of tokens. So, it's not dynamically adapting to the number of tokens. 

So, to address this limitation, another strategy is top <math><mi>P</mi></math>, where we can dynamically set the number of tokens to sample from. And in this case, we would sample from the minimum set of tokens whose cumulative of probability is greater than or equal to <math><mi>P</mi></math>. So, in this case, if we set <math><mi>P</mi></math> to be 0.75 , we just add the probabilities starting from the most probable token. So, that's flowers at 0.5 , and then we add  <math><mi>0.23,0.05</mi></math>, and now we've hit the threshold of 0.75 . 

<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-16_at_7.42.55%E2%80%AFPM.png" width="80%"/>

<img src="/deeplearningai/google-cloud-vertex-ai-embeddings/images/Screenshot_2024-03-16_at_7.43.23%E2%80%AFPM.png" width="80%"/>

So, we would sample from these three tokens alone. So, you don't need to set all of these different values, but if you were to set all of them, this is how they all work together.

- Top p: sample the minimum set of tokens whose probabilities add up to probability `p` or greater.
- The default value for `top_p` is `0.95`.
- If you want to adjust `top_p` and `top_k` and see different results, remember to set `temperature` to be greater than zero, otherwise the model will always choose the token with the highest probability.
- 
```python
top_p = 0.2
```


```python
prompt = "Write an advertisement for jackets \
that involves blue elephants and avocados."
```


```python
response = generation_model.predict(
    prompt=prompt, 
    temperature=0.9, 
    top_p=top_p,
)
```


```python
print(f"[top_p = {top_p}]")
print(response.text)
```

First, the tokens are filtered by top K, and from those top K, they're further filtered by top P. And then finally, the output token is selected using temperature sampling. And that's how we arrive at the final output token. So, let's jump into the notebook and try and set some of these values. So, first we'll start off by setting a top $P$ value of 0.2 . And note that by default, the top $P$ value is going to be set at 0.95. And this parameter can take values between 0 and 1 . So, nere is a fun prompt. Let's ask for an advertisement about jackets that involves blue elephants and avocados, two of my favorite things. So, we can call the generation model predict function again. And this time, we'll pass in the prompt. We'll also pass in a temperature value, let's try something like 0.9 , and then we'll also pass in top P. And note that temperature by default at zero does result in a deterministic response.

It's greedy decoding, so the most likely token will be selected at each timestamp. So, if you want to play around with top $\mathrm{P}$ and top , just set the temperature value to something ther than zero. So, we can print out the response here and see what we get. And here, is an advertisement introducing this new blue elephant avocado jacket. So lastly, let's just see what it looks like to set top P and top K. So let's set a top K to 20. And by default, top K is going to be set to 40 . And this parameter takes values between one and 40. So, we'll half that default value. And then, we'll also set top P so we can set all three of these parameters we just learned about. And we'll use the exact same prompt as before, we'll just keep it as write an advertisement for jackets that involves blue elephants and avocados.

And this time, when we call the predict function on our generation model, we'll pass in the prompt, the temperature value, the top $\mathrm{k}$ value, and the top $p$ value. And just as a reminder, this means that the output tokens will be first filtered by the top $k$ tokens, then further filtered by top $p$, and lastly, the response tokens will be selected with temperature sampling. So here, we've got a response here, and we can see that it is a little different from the one we saw earlier. So, I encourage you to try out some different values for top <math><mi>P</mi></math> top $\mathrm{k}$, and temperature, and also try out some different prompts and see what kinds of interesting responses or use cases or behaviors you can get these arge language models to take on. So, just as a quick recap of the syntax we just learned, again, we've been importing this text generation nodel, and then we loaded this text bison nodel, and we define a prompt, which is the input text to our model. And when we call predict, we can, in addition to passing in a prompt, also pass in a value for temperature, top <math><mi>K</mi></math> and top P. So now that you know a little bit about how to use these models for text generation, I encourage you to jump into the notebook, try out some different temperature, top P and top $\mathrm{K}$ values, and also experiment with some different prompts. And when you're ready, we'll take what you've learned

It's greedy decoding, so the most likely token will be selected at each timestamp. So, if you want to play around with top <math><mi>P</mi></math> and top <math><mi>K</mi></math>, just set the temperature value to something ther than zero. So, we can print out the response here and see what we get. And here, is an advertisement introducing this new blue elephant avocado jacket. So lastly, let's just see what it looks like to set top P and top K. So let's set a top K to 20. And by default, top K is going to be set to 40 . And this parameter takes values between one and 40. So, we'll half that default value. And then, we'll also set top P so we can set all three of these parameters we just learned about. And we'll use the exact same prompt as before, we'll just keep it as write an advertisement for jackets that involves blue elephants and avocados.

And this time, when we call the predict function on our generation model, we'll pass in the prompt, the temperature value, the top <math><mi>k</mi></math>value, and the top <math><mi>P</mi></math> value. And just as a reminder, this means that the output tokens will be first filtered by the top $k$ tokens, then further filtered by top <math><mi>P</mi></math>, and lastly, the response tokens will be selected with temperature sampling. So here, we've got a response here, and we can see that it is a little different from the one we saw earlier. 


- The default value for `top_k` is `40`.
- You can set `top_k` to values between `1` and `40`.
- The decoding strategy applies `top_k`, then `top_p`, then `temperature` (in that order).


```python
top_k = 20
top_p = 0.7
```


```python
response = generation_model.predict(
    prompt=prompt, 
    temperature=0.9, 
    top_k=top_k,
    top_p=top_p,
)
```


```python
print(f"[top_p = {top_p}]")
print(response.text)
```


So, I encourage you to try out some different values for top <math><mi>P</mi></math>, top <math><mi>K</mi></math>, and temperature, and also try out some different prompts and see what kinds of interesting responses or use cases or behaviors you can get these arge language models to take on. So, just as a quick recap of the syntax we just learned, again, we've been importing this text generation nodel, and then we loaded this text bison nodel, and we define a prompt, which is the input text to our model. And when we call predict, we can, in addition to passing in a prompt, also pass in a value for temperature, top <math><mi>K</mi></math> and top P. So now that you know a little bit about how to use these models for text generation, I encourage you to jump into the notebook, try out some different temperature, top P and top <math><mi>K</mi></math> values, and also experiment with some different prompts. And when you're ready, we'll take what you've learned