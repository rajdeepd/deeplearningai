---
layout: default
title: Generative AI project lifecycle
nav_order: 9
description: "Generative AI project lifecycle"
has_children: false
parent: Week1
grand_parent: Coursera - GenAI with LLMs 
---

You will learn the Gen AI project lifecycle from inception to launch. It will talk about the potential problems and the infrastructure required to deploy the application.
Diagram for the overall lifecycle is shown below

### Define the use case


The most important step in any project is to define the scope as accurately and narrowly as you can. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-19_at_6.29.08_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-19_at_6.30.11_PM.png" width="80%" />


As you've seen in this course so far, LLMs are capable of carrying out many tasks, but their abilities depend strongly on the size and architecture of the model. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-19_at_6.30.32_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-19_at_6.30.37_PM.png" width="80%" />

You should think about what function the LLM will have in your specific application.

Do you need the model to be able to carry out many different tasks, including long-form text generation or with a high degree of capability, or is the task much more specific like named entity recognition so that your model only needs to be good at one thing. 




As you'll see in the rest of the course, getting really specific about what you need your model to do can save you time and perhaps more importantly, compute cost. 

### Select the model

Once you're happy, and you've scoped your model requirements enough to begin development. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-19_at_6.48.31_PM.png" width="80%" />


Your first decision will be whether to train your own model from scratch or work with an existing base model. In general, you'll start with an existing model, although there are some cases where you may find it necessary to train a model from scratch. 



You'll learn more about the considerations behind this decision later this week, as well as some rules of thumb to help you estimate the feasibility of training your own model. 

## Adapt and align the model

With your model in hand, the next step is to assess its performance and carry out additional training if needed for your application. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-19_at_6.48.50_PM.png" width="80%" />

As you saw earlier this week, prompt engineering can sometimes be enough to get your model to perform well, so you'll likely start by trying in-context learning, using examples suited to your task and use case. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-19_at_6.48.57_PM.png" width="80%" />

There are still cases, however, where the model may not perform as well as you need, even with one or a few short inference, and in that case, you can try fine-tuning your model.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-19_at_7.02.57_PM.png" width="80%" />

This supervised learning process will be covered in detail in Week 2, and you'll get a chance to try fine tuning a model yourself in the Week 2 lab. As models become more capable, it's becoming increasingly important to ensure that they behave well and in a way that is aligned with human preferences in deployment. 

In Week 3, you'll learn about an additional fine-tuning technique called reinforcement learning with human feedback, which can help to make sure that your model behaves well. An important aspect of all of these techniques is evaluation. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-19_at_7.03.18_PM.png" width="80%" />

Next week, you will explore some metrics and benchmarks that can be used to determine how well your model is performing or how well aligned it is to your preferences.

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-19_at_7.03.28_PM.png" width="80%" />


Note that this adapt and aligned stage of app development can be highly iterative. You may start by trying prompt engineering and evaluating the outputs, then using fine tuning to improve performance and then revisiting and evaluating prompt engineering one more time to get the performance that you need. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-20_at_8.32.22_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-20_at_8.32.28_PM.png" width="80%" />



Finally, when you've got a model that is meeting your performance needs and is well aligned, you can deploy it into your infrastructure and integrate it with your application. At this stage, an important step is to optimize your model for deployment. This can ensure that you're making the best use of your compute resources and providing the best possible experience for the users of your application. The last but very important step is to consider any additional infrastructure that your application will require to work well. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-20_at_8.32.33_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-20_at_8.32.38_PM.png" width="80%" />

There are some fundamental limitations of LLMs that can be difficult to overcome through training alone like their tendency to invent information when they don't know an answer, or their limited ability to carry out complex reasoning and mathematics. 

In the last part of this course, you'll learn some powerful techniques that you can use to overcome these limitations. I know there's a lot to think about here, but don't worry about taking it all in right now. You'll see this visual over and over again during the course as you explore the details of each stage

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-20_at_8.32.46_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-07-20_at_8.32.51_PM.png" width="80%" />

In Week 3, you'll learn about an additional fine-tuning technique called reinforcement learning with human feedback, which can help to make sure that your model behaves well. An important aspect of all of these techniques is evaluation. Next week, you will explore some metrics and benchmarks that can be used to determine how well your model is performing or how well aligned it is to your preferences. 
​

