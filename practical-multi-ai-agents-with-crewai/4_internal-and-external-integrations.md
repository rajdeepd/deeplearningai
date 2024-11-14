---
layout: default
title: 4. Internal and External Integrations
nav_order: 4
description: " Automated Project: Planning, Estimation, and Allocation"
parent: Practical Multi AI Agents with crewAI
has_children: true
---

In this lesson you're going to learn about integrations.
And this is super important for all of your Al agent automations and apps, you're going to need to pull
or push information from either internal or external systems.
And in this lesson, you are going to learn everything there is to know about how to build these integrations
and how to make your agents to be able to call with these systems.
Let's dive into the lesson.
You already learned that you're crew is your agents, your tasks and your tools.

<img src="./images/Screenshot 2024-11-11 at 12.23.38 PM.png"/>
And there's a few different moments during a crew execution where you might want to talk with either internal or external systems.
Sometimes you might want to talk with the systems prior to crew starting, so that we can have data that you can pass on to this crew.
Other times you might want to call it once that the crew is done processing.
Those two you are less of a problem because by them is just regular code.
But the problem is, sometimes you want your tools to be able to call external or internal systems.
So, these tools can call other applications or cloud services.
Sometimes they can call databases or internal apps that you might have.

<img src="./images/Screenshot 2024-11-11 at 12.24.23 PM.png"/>
Some examples might be a tool that searches the internet, or check a calendar, or reply an email.
Or if they're doing something internally, they might be doing
a RAG search on an existing embeddings database, or doing a SQL query, or even triggering a side effect.
The thing is, you want to be mindful about allowing your agents to be able to use these systems, and sometimes you might
want your agents to be able to do some custom things like writing code.

<img src="./images/Screenshot 2024-11-11 at 12.24.32 PM.png"/> 

And you're going to learn and how to allow your agents to do so, and what that will look like in one of four lessons.
<!-- 2:00 -->
Now let's talk about one of the great things for internal and external systems.
And that is the real-time reaction
and the ability for these agents to self heal.
What I mean by that is that your agents are calling these tools.

<img src="./images/Screenshot 2024-11-11 at 12.28.27 PM.png"/> 

If those tools fail for whatever reason, maybe it's a change in the system, either externally or internally, your agents will be able to pick up on their change and try to do it again, but differently.
So let's talk about an example.
Let's say that we have a crew that is doing analysis, and it's trying to leverage external and internal data to do it.
So we start with an initial task where an agent's trying to extract data sources from data on a company.
So it's doing research around the company, its industry and everything about it.
Then it's going to cross that with internal data of that company.
So whatever existing reports that you might already have on that
same company. Then it's going to try to pull together through company records based on everything that it learned about the given company.
Sometimes that might be missing information.
So if it does find that information is missing, it's going to try to find missing data points by doing an embedding searching using a RAG tool.
Here you can see how these agents can go so beyond that, for a regular simple RAG search or an enrichment pipeline, by being able to tap into so many different sources of data externally, internally, using embeddings and so much more.
This unlocks a lot of potential in use cases, and we're going to build one ourselves right now.
So now let's jump into the Jupyter notebook and build this crew ourselves. Let's get our hands dirty and integrate with external tools.
This is going to be a lot of fun. So stick around and I see you in the next step.
In a second.