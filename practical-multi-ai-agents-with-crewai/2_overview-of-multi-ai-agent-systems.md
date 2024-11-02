---
layout: default
title: Overview of Multi-Agent AI System
nav_order: 2
description: "Overview of Multi-Agent AI System"
parent: Practical Multi AI Agents with crewAI
---

# Multi Agent AI Systems Overview

This lesson provides an overview of multi-AI agent systems, exploring the building blocks of AI agents, including tasks, crews, caching, memory, and guardrails. 

<img src="./images/Screenshot 2024-10-31 at 8.50.54 PM.png" width="100%"/>

The course will cover practical applications of multi-agentic automation, allowing learners to build projects and understand various use cases. Examples include automated project planning, project monitoring, lead qualification and scoring, support data analysis and reporting, and custom content creation at scale. 

<img src="./images/Screenshot 2024-10-31 at 8.51.58 PM.png" width="100%"/>

<img src="./images/Screenshot 2024-10-31 at 8.58.58 PM.png" width="=100%"/>

To illustrate the power of these systems, the lesson showcases an example of generating sales materials for a meeting with the CEO of Zendesk. This involves researching the CEO and his company, then automatically creating a personalized landing page/PDF highlighting the relevance of CrewAI from Zendesk's perspective. 

<img src="./images/Screenshot 2024-10-31 at 8.59.02 PM.png" width="100%"/>

<img src="./images/Screenshot 2024-10-31 at 8.59.06 PM.png" width="100%"/>

The lesson emphasizes the potential of these agents to perform research and produce comprehensive reports for various applications, including operational automation, sales, marketing, and code generation.

<img src="./images/Screenshot 2024-10-31 at 8.59.12 PM.png" width="100%"/>

-- 2:48--

The typical workflow for multi-agent automation follows a general pattern, often resembling a long tail distribution.  

### Data Extraction

It usually begins with extracting data from existing systems such as ERPs, CRMs, or databases.


### Research

This data then undergoes a research phase, potentially involving document analysis, internet searches, or querying other systems.

### Analysis

Next, the data is analyzed. T

<img src="./images/Screenshot 2024-11-01 at 7.01.49 PM.png" width="100%"/>

This might involve comparisons, extractions, or inferring new information.

### Summarization

A summarization process typically follows, generating learnings, charts, or executive summaries as show in the figure below.

<img src="./images/Screenshot 2024-11-01 at 7.02.02 PM.png" width="100%"/>

### Reporting

Finally, the process concludes with reporting, often delivered as a PDF, JSON, or markdown file, which can then be integrated into another system.  

<img src="./images/Screenshot 2024-11-01 at 7.02.19 PM.png" width="100%"/>

Regardless of the specific industry—sales, marketing, HR, etc.—most use cases involve some combination of research, analysis, summarization, and reporting.  While not every use case follows this exact sequence, it represents the majority of observed applications. 


Some companies push boundaries with innovative applications using video and image models.

### Introduction to Multi-Agent Systems and CrewAI

For those new to multi-agent systems and CrewAI, we will explore their nature and construction. 


<img src="./images/Screenshot 2024-11-01 at 7.05.52 PM.png" width="100%"/>

Traditional apps, built by engineers, are strongly typed, meaning the input data and its transformations are well-defined.  A typical example is a lead form processing system, where specific input triggers predetermined automations and outputs.  AI apps, however, differ significantly.


<img src="./images/Screenshot 2024-11-01 at 7.06.00 PM.png" width="100%"/>


<!--5:09--->

If you think about regular apps, it usually is very strong typed.
What I mean by that,is that you have a verygood understanding of what is the data that's coming into your application, and you also have a great understanding of what are the transformations that this data is going to go through in order to give you your expected output.

<img src="./images/Screenshot%202024-11-01%20at%209.54.37 PM.png"/>
So a good example of this would be a system where you're getting inputs from a lead form,and you will understand that you have a series of conditions that depending
on these answers, you're going to have a series of automations or outputs.

But then if you look at these new apps, what I'm calling here, Al apps, they're extremely different. Because now they're way more fuzzy.
That means that you don't have a good understanding on what are the data that is coming in to this.

<img src="./images/Screenshot 2024-11-01 at 9.55.37 PM.png"/> 
For example, if you think about ChatGPT, you don't know if the text that user's putting there is a recipe
or a PhD thesis or whatever it might be, it's fuzzy.
And then this data goes through a model that is a black box, and that then goes into a fuzzy output because you don't
know what the output will be.

<img src="./images/Screenshot 2024-11-01 at 9.55.48 PM.png"/>
It really depend heavily on the input and the model.
1 So you can think about Multi-Al agents as a kind of Al app that is way more fuzzy. But because of that,
It allows you to build automations that were just not possible before,
because now you don't need to treat for every single edge case out there.

You can basically let your agents decide on real time how they're going to react to specific data and deciding what tools to use in order to achieve the task that you want it to do.
When you look at these agents, what are their anatomy?
Well, it starts pretty simple.

<img src="./images/Screenshot 2024-11-01 at 9.56.02 PM.png"/>

You have an LLM in the center, and this LLM can have access to some tools.
And once you give this LLM a task, it's going to find a way to use these tools in order to provide you a final answer. And when you go one step higher, what you see is actually this multi-agent systems where now you just don't have one agent, but you have two, three or many more.
And now these agents can not only use tools themselves,
but they can delegated work to each other, and they can ask questions to each other 3 in order to accomplish whatever is the final outcome that you want it to.
And it starts usually pretty simple, but once that you start to bring this automation into a production setting,
what do you realize is there's so many needs you're going to have to learn that you're going to need a caching layer.
So whatever tools your agents are using, they're not consuming and necessary credits.
<img src="./images/Screenshot 2024-11-01 at 10.02.39 PM.png"/>

Using these tools over and over and over again.

<!-- 755  -->
You also want to make sure that they have a memory layer, so they remember what they have done in the past, and they share their memory with each other.
So if they ever have come across this same data point again, they will remember and how they handle it in the last time.
There's also training data, something that we're going to talk in this specific lesson.
I'm so excited about that. 
There's also guardrails and how you're going to protect these agents from going into crazy hallucinations and so much more.

<img src="./images/Screenshot 2024-11-01 at 10.03.04 PM.png"/>

Not only all these features, but once that you have these agents working together, you really to think through how you want to orchestrate them.
Sometimes you just want them to do their work sequentially.
Other times, you want to have a manager agent that is going to delegate work and review the output, but you can go crazy on that.

<img src="./images/Screenshot%202024-11-01%20at%2010.03.27 PM.png"/>

So you can have a hybrid approach where some tasks are going to be performed in parallel, and other tasks are going to wait for multiple tasks to finish before moving on.
You're going to create an example like this as well.
Others are going to be complete in parallel and others may be completely asynchronous.
<img src="./images/Screenshot 2024-11-02 at 11.21.43 AM.png"/>

So there are so many different use cases. You can get even more complex if you want to by doing multi-crews. You're going to use a feature that we call flows.
Where you're able to hook one crew result with another.

<img src="./images/Screenshot 2024-11-02 at 11.21.59 AM.png"/
<!-- 930 -->

So you just learned about Al agents, how they work, what are their anatomy, and how you can put them to work together. Well, what are the main building blocks for building these multi-Al agent systems?

<img src="./images/Screenshot 2024-11-02 at 11.22.26 AM.png"/>
Well, everything starts with agents. But then you also need to make sure that you have the tasks.


<img src="./images/Screenshot 2024-11-02 at 11.23.31 AM.png"/>
In this use case, you can see that we have more tasks than agents.
And that is not a problem because one agent could be doing multiple tasks.
And we're going to see some examples of that.
In order for these agents to be able to accomplish these tasks,
we're going to need to give them tools.

<img src="./images/Screenshot 2024-11-02 at 11.23.38 AM.png"/>

4 So you can either give your agents tools, so they can use data
when performing any tasks, or you can give your tasks tools,
so your agents know what tools to use in order to accomplish that task.
Once that you have that, you basically have a crew.
A crew is a combination of multi agents and their tasks.

And once that you have all of these agents and tasks,
CrewAl comes in time and gives you all the features that you need to run this 3 things in production by adding guardrails to avoid your agents to hallucinate, 3 but also testing so that you can measure the quality of your agents and tasks.
Also allows delegations where your agents can automatically delegate and ask questions to each other, and then training data so you can train these agents even further in the memory.

<img src="./images/Screenshot 2024-11-02 at 11.22.47 AM.png"/>
These agents get it better over time.

There's so much that we're going to talk about make sure to stick around.
Let's look at these agents and tasks. Every agent and CrewAl needs to have a role a goal and a backstory.

<img src="./images/Screenshot 2024-11-02 at 11.42.27 AM.png"/>

Every task needs to have a description, an expected output and an agent.
So now these agents are actually defined as Yaml files.

<img src="./images/Screenshot 2024-11-02 at 11.42.07 AM.png"/>

You're going to go over it on our lessons where you can easily
see how these agents have their role that goes and their backstory set and their tasks as well with their descriptions and their expected outputs.
This makes extra easy for non-technical people
to be able to contribute to these agents and tasks by just
having to update Yaml files instead of having to update any code.

So now that we know that we can create agents and tasks using Yaml files, why do we get our hands dirty?
We built our first crew. This is going to be so exciting.

Let's go to our next step where we're going to dive into a Jupyter notebook.
And we're going to put together our first crew together.
I'll see you there in a second.

<!-- 11:01-->
