---
layout: default
title: Reward Model
nav_order: 4
description: "Reward Model"
has_children: false
parent: Week3
grand_parent: Coursera - GenAI with LLMs 
tags: [MathJax, Mathematic]
mathjax: true
---
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
# Reward Model


At this stage, you have everything you need to train the reward model. While it has taken a fair amount of human effort to get to this point, by the time you're done training the reward model, you won't need to include any more humans in the loop. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-09-22_at_3.58.57_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-09-22_at_3.59.34_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-09-22_at_4.00.07_PM.png" width="80%" />

Instead, the reward model will effectively take place off the human labeler and automatically choose the preferred completion during the oral HF process. This reward model is usually also a language model. For example, a bird that is trained using supervised learning methods on the pairwise comparison data that you prepared from the human labelers assessment off the prompts. For a given prompt $$X$$, the reward model learns to favor the human-preferred completion $$y_ j$$, while minimizing the lock sigmoid off the reward difference, $$rj - r_k$$.

As you saw on the last slide, the human-preferred option is always the first one labeled $$y_j$$. Once the model has been trained on the human rank prompt-completion pairs, you can use the reward model as a binary classifier to provide a set of logics across the positive and negative classes. Logics are the unnormalized model outputs before applying any activation function. Let's say you want to detoxify your LLM, and the reward model needs to identify if the completion contains hate speech. In this case, the two classes would be notate, the positive class that you ultimately want to optimize for and hate the negative class you want to avoid. The largest value of the positive class is what you use as the reward value in LLHF. Just to remind you, if you apply a Softmax function to the logits, you will get the probabilities. The example here shows a good reward for non-toxic completion and the second example shows a bad reward being given for toxic completion. 

<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-09-22_at_10.52.00_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-09-22_at_10.52.13_PM.png" width="80%" />
<img src="/deeplearningai/generative-ai-with-llms/images/Screenshot_2023-09-22_at_10.52.30_PM.png" width="80%" />

I know this lesson has covered a lot so far. But at this point, you have a powerful tool in this reward model for aligning your LLM. The next step is to explore how the reward model gets used in the reinforcement learning process to train your human-aligned LLM. Join me in the next video to see how this works.