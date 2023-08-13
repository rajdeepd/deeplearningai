---
layout: default
title: Aligning models with human values
nav_order: 2
description: "Aligning models with human values"
has_children: false
parent: Week3
grand_parent: Coursera - GenAI with LLMs 
---

# Aligning models with human values

## Generative Al project lifecycle - Fine tuning

Let's come back to the Generative AI project life cycle. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-13_at_10.10.17_AM.png" width="80%"/>

Last week, you looked closely at a technique called fine-tuning. The goal of fine-tuning with instructions, including path methods, is to further train your models so that they better understand human like prompts and generate more human-like responses. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-13_at_10.10.43_AM.png" width="80%"/>

This can improve a model's performance substantially over the original pre-trained based version, and lead to more natural sounding language. However, natural sounding human language brings a new set of challenges.





## Models behaving badly

By now, you've probably seen plenty of headlines about large language models behaving badly. Issues include models using toxic language in their completions, replying in combative and aggressive voices, and providing detailed information about dangerous topics. 

---

Models behave bady because of the following reasons

- Toxic language
- Aggressive responses
- Providing dangerous information

---

These problems exist because large models are trained on vast amounts of texts data from the Internet where such language appears frequently. Here are some examples of models behaving badly. Let's assume you want your LLM to tell you knock, knock, joke, and the models responses just clap, clap. 


<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-13_at_10.11.52_AM.png" width="80%"/>

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-13_at_10.11.58_AM.png" width="80%"/>

While funny in its own way, it's not really what you were looking for. The completion here is not a helpful answer for the given task. Similarly, the LLM might give misleading or simply incorrect answers. If you ask the LLM about the disproven Ps of health advice like coughing to stop a heart attack, the model should refute this story. Instead, the model might give a confident and totally incorrect response, definitely not the truthful and honest answer a person is seeking. Also, the LLM shouldn't create harmful completions, such as being offensive, discriminatory, or eliciting criminal behavior, as shown here, when you ask the model how to hack your neighbor's WiFi and it answers with a valid strategy. Ideally, it would provide an answer that does not lead to harm. These important human values, helpfulness, honesty, and harmlessness are sometimes collectively called HHH, and are a set of principles that guide developers in the responsible use of AI. 


## Generative Al project lifecycle - Human Feedback

Additional fine-tuning with human feedback helps to better align models with human preferences and to increase the helpfulness, honesty, and harmlessness of the completions. 


<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-08-13_at_10.12.06_AM.png" width="80%"/>

This further training can also help to decrease the toxicity, often models responses and reduce the generation of incorrect information. In this lesson, you'll learn how to align models using feedback from humans. Join me in the next video to get started.
