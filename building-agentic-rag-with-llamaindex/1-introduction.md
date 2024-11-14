---
layout: default
title: 0. Introduction
nav_order: 1
description: "Computational challenges of training LLMs"
has_children: false
parent:  Building Agentic RAG with Llamaindex
---
Welcome to Building Agentic RAG with Llamalndex.
I'm joined by Jerry Liu, who is co-founder and CEO at Llamalndex and the instructor for this course. Thanks, Andrew. I'm super excited to be here with you.
In this course, you learn about a agentic RAG, a framework
to help you build research agents capable of events
to use reasoning and decision making over your data.
For example, one of you have a set of research papers on some topic and you want to pull out the parts relevant to a question you want to ask, and you get a synthesis of what the papers say. This is a complex requests that require multiple steps of processing. Further, various steps of processing, like maybe identifying a theme for one paper, but also change the later steps that are needed, like retrieving additional information from other papers about that theme. In comparison, the standard RAG pipeline, which is very popular, is mostly good for simpler questions over a small set of documents and works by retrieving some context, sticking that into the prompt and then just calling a single time o get a response.
This course will take the idea of chatting over your data to the next level, and show you how to build an autonomous research agent.
'ou'll learn a progression of reasoning ingredients
o building a full agent. First, routing.
Ne add decision making to route requests to multiple tools. Next, tool use. Where you create an interface for agents, to selected tool, as well as generate the right arguments for that tool. And then finally, multi-step reasoning with tool use.
We'll use LLM to perform multi-step reasoning with a range of tools for retaining memory throughout that process.
You will learn how to effectively interact with an agent and use its capability for detailed control and oversight.
This will allow you to not only create a higher level research assistant over your RAG pipelines, but also give you more effective ways to guide its actions. Specifically, you'll learn how to ensure debug ability of the LLM.
We'll look at how to step through
what your agent is doing and how to use that to improve your agent.
One additional very powerful tool is to let the user optionally inject guidance at intermediate steps.
For example, if you see us searching the wrong document,
a little nudge to search a different document from human input,
much like an experienced manager, that's you, giving a more junior employee a nudge to consider a new piece of information can give much better performance.
Many people have worked to create this course.


I'd like to thank from Llamalndex, Logan Markewich and Andrei Fajardo. From DeepLearning.Al, Diala Ezzeddine also contributed to this course. In the first lesson, you will build a router over a single document that can handle both question answering as well as summarization. That sounds great.
Let's go on to the next video and get started.