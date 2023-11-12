---
layout: default
title: Semantic Kernel AI kitchen
nav_order: 2
description: ".."
has_children: false
parent:  Coursera - Microsoft Semantic Kernel
---

# 🔥 Get a kernel ready

So there's a ton of AI content out there. You're hoping this is the one that's gonna help you figure this stuff out. You know it has to be hands-on, otherwise it's just blah blah blah. That's why we're here. Symantec Kernel has been made for large-scale enterprises, but it's also available to you. Someone who's trying to figure things out. and we're going to jump into it in the spirit of cooking because as you know you can't cook if you don't have a kitchen. Spider-Kernel is a kitchen for your AI recipes so you might be able to make meals for yourself, for your friends, your family, for customers that will you know make your life a little bit better. So let's jump in, get started, come along. So welcome to your kitchen. Well we're not going to get too deep in the kitchen yet, so sit tight. We're going to do a quick overview, kind of like when you buy a big machine, you get it with a manual. As a manual, we want to help you understand what Symantec Kernel means. Number one, it's a kernel. What is a kernel? 



A kernel is something that is at the center of a computational system that is really important. So the kernel is the essential glue, the orchestrator of different tasks. So super important.

<img src="/deeplearningai/microsoft-semantic-kernel/images/Screenshot_2023-10-24_at_6.24.51_PM.png" />


Secondly, it's semantic. Why is it semantic? Why did we use that word? It's because Sam Scalace, the person who got this all started, said this kind of computation that uses AI models and memory models that use language is inherently semantic, not syntactic, not brittle, more flexible. 

<img src="/deeplearningai/microsoft-semantic-kernel/images/Screenshot_2023-10-24_at_6.25.11_PM.png" />

So semantic was a word we used for semantic kernel. And it's got a job. 

The job is to be your amazing kitchen to take any kind of sort of test kitchen recipes and bring it all the way to production. Not just production to serve like five people at your house, but to serve five million people all over the world. So let's jump into Notebook. So every Notebook is a bit daunting but as you know, you want to bring in things into the world as import and you give it a short name. I'm going to import semantic kernel. I'm going to make a kernel and then I'm going to want to connect the kernel with some model. I'm use this syntax because I'm calling the OpenAI settings from dotenv and I'm going to add to the kernel a text completion service that we're going to give it a label OpenAI. We're going to make sure we clarify that it is a. OpenAI chat completion. 

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion

kernel = sk.Kernel()

useAzureOpenAI = False

if useAzureOpenAI:
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel.add_text_completion_service("azureopenai", AzureChatCompletion(deployment, endpoint, api_key))
else:
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_text_completion_service("openai", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id))

print("You made a kernel!")
```

    You made a kernel!

## Semantic Kernel is Open-Source

In the open-source world of SDKs for LLMs, Semantic Kernel is the option that is the most enterprise-y and "no frills." Many of the available code examples demonstrate how to dock into the universe of Azure Al components for things like Al guardrails, prompt tuning, and vector databases. .NET/C\\# and Python are supported with branches available in TypeScript and Java.