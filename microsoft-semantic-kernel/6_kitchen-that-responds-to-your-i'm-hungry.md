---
layout: default
title: A kitchen that responds to your “I’m hungry”
nav_order: 6
description: ".."
has_children: false
parent:  Coursera - Microsoft Semantic Kernel
---
 
We're going to imagine a world where you just say I'm hungry and the LLM is able to complete that meal, use the plugins you've created and voila. 
Think of this small business owner. 
There are two buckets every business owner has to deal with 
The bucket of time is leaking. The bucket of money is leaking. 
Do they have time to think about how to solve their business problems? 
 
Now you have the tools to be able to give them interesting solutions, 
but at the same time, if they could just say, 
I wish I could do X and just have the AI help the business owner with as little effort as possible. 
For instance, let's say that the wish were something like, I wish I was $10 richer. I will need to blank. 
You know about the completion engine now, right? 
It'll complete it, but it's going to hallucinate because it's making stuff 
up. 

You use retrieval augmented generation to find similar things from your knowledge base. You might have different things that plugins could do, 
like write marketing copy or send an email. 
You can make plugins that do all kinds of things, native or semantic. 
And once you do that, what happens is each 
plugin will have a description of some form. 
And all you have to do is use that similarity engine to find the different tools in your tool bucket. 

That's what happens when you have the similarity engine. You can magnetize all your plugins. 
If you had 2000 plugins, you as a human do not want to like use every plugin yourself. 
You want the AI to find the right plugin. So the third step would be finding the relevant plugins and then use those to be used in the completion response. 

Remember, VAT of plugin, pull out the similar ones, use it for completion, push them to the kernel. 
When you have this thing called a planner, the planner does that for you. Luckily in Semantec Kernel, we love plugins. 
Now you're learning that we love planners because if you got a lot of plugins, you'll need a planner. 
Lets do our quick inventory exercise, which you know I like, I'm a Wes Anderson fan. 

## Let's make a kernel. 

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAITextEmbedding
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from IPython.display import display, Markdown

kernel = sk.Kernel()

useAzureOpenAI = False

if useAzureOpenAI:
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel.add_text_completion_service("azureopenaicompletion", AzureChatCompletion(deployment, endpoint, api_key))
    kernel.add_text_embedding_generation_service("azureopenaiembedding", AzureTextEmbedding("text-embedding-ada-002", api_key, endpoint))
else:
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_text_completion_service("openaicompletion", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id))
    kernel.add_text_embedding_generation_service("openaiembedding", OpenAITextEmbedding("text-embedding-ada-002", api_key, org_id))
print("I did it boss!")
```

We want to have a VAT of plugins and then find the right plugin to fit the goal. So how do we do that? There's different kinds of planners. Remember the planners, different kind of planners. The reality is there is a super simple planner,and call it the Action Planner. 

In the Action Planner, you create it from the kernel. You give it a bunch of skills, plugins. If you notice Semantic Kernel has a complex where there's skills (but just call them plugins). 
But what I'm doing right now is I'm adding the tools for the kernel to do math, to read files, to tell 
the time and to play with text. 


**Note**: You can find more about the predefined plugins used below [here](https://learn.microsoft.com/en-us/semantic-kernel/ai-orchestration/out-of-the-box-plugins?tabs=Csharp).





 
Next we're going to do is we're going to use the planner to create a one, mind you, a one function, a single function that's gonna be pulled out of a vat of plugins to use. 

What it's doing is it's taking this ask, and this is basically looking through all the available plugins it has available to it, skills, functions, et cetera, okay? What I'm gonna do is I'm gonna ask it to tell me what that function is, right? 

All right, let's add some more print statements around this. Because programming is pretty abstract unless it tells you what you're doing. 
So let's run this. 

```python
from semantic_kernel.planning import ActionPlanner

planner = ActionPlanner(kernel)

from semantic_kernel.core_skills import FileIOSkill, MathSkill, TextSkill, TimeSkill
kernel.import_skill(MathSkill(), "math")
kernel.import_skill(FileIOSkill(), "fileIO")
kernel.import_skill(TimeSkill(), "time")
kernel.import_skill(TextSkill(), "text")

print("Adding the tools for the kernel to do math, to read/write files, to tell the time, and to play with text.")
```

Its finding the most similar function available to get that done. What did it do? Wow, it knew that if I'm trying to get the sum of two numbers, it found in the math plugin, the addition function. How did it find it? Well, remember we made a description for each function. 
It's just comparing this question into the function description. Not a surprise, is it? 

```python
ask = "What is the sum of 110 and 990?"

print(f"🧲 Finding the most similar function available to get that done...")
plan = await planner.create_plan_async(goal=ask)
print(f"🧲 The best single function to use is `{plan._skill_name}.{plan._function.name}`")

```

Like, let's say like a, what is today? It's gonna look through the available plugins and it found in the time plugin, the today function. 

```python
ask = "What is today?"
print(f"🧲 Finding the most similar function available to get that done...")
plan = await planner.create_plan_async(goal=ask)
print(f"🧲 The best single function to use is `{plan._skill_name}.{plan._function.name}`")
```

Do you see how that's working? It, you know, if you totally do something very complex, 
like what is the way to get to San Jose when the traffic is really bad? 

Now, this might require many plugins to work in concert, but as you can see, it's like, no, I really can't do that, boss, you know? 
 
So for simple things, how do I write the word text to a file? It's probably going to find in the file IO skill, it 
found the write function. 

```python
ask = "How do I write the word 'text' to a file?"
print(f"🧲 Finding the most similar function available to get that done...")
plan = await planner.create_plan_async(goal=ask)
print(f"🧲 The best single function to use is `{plan._skill_name}.{plan._function.name}`")
```

Pretty cool, right? Again, it's very limited. It's no insult to you, computer. It's just not that smart. 
But when you can do that, a simple planner, you can imagine a planner that is much more powerful. 
And so the action planner is good for like a very basic searching through, find just one function. 
But what if I wanted to do a multi-step plan that's automatically generated, right? 
Let's do that. 
So what we're gonna do is we're going to pull in the sequential planner. The sequential planner is our gen two planner. There's a gen three planner that is, it's been ported from C-sharp. so it'll be coming in the repo shortly. Again, this is all open source, so you can get access to the latest and greatest as it comes out.


And all I'm gonna do is I'm going to bring in the literate friend plugin that I have. 
The literate friend plugin has a few functions. One, it can write poetry, it can translate, but I'm gonna hold onto that. 

**Note**: The next two cells will *sometimes return an error*. The LLM response is variable and at times can't be successfully parsed by the planner or the LLM will make up new functions.  If this happens, try resetting the jupyter notebook kernel and running it again.

```python
from semantic_kernel.planning import SequentialPlanner
from semantic_kernel.core_skills.text_skill import TextSkill
from semantic_kernel.planning.sequential_planner.sequential_planner_config import SequentialPlannerConfig

plugins_directory = "./plugins-sk"
writer_plugin = kernel.import_semantic_skill_from_directory(plugins_directory, "LiterateFriend")

# create an instance of sequential planner, and exclude the TextSkill from the list of functions that it can use.
# (excluding functions that ActionPlanner imports to the kernel instance above - it uses 'this' as skillName)
planner = SequentialPlanner(kernel, SequentialPlannerConfig(excluded_skills=["this"]))

ask = """
Tomorrow is Valentine's day. I need to come up with a poem. Translate the poem to French.
"""

plan = await planner.create_plan_async(goal=ask)

result = await plan.invoke_async()

for index, step in enumerate(plan._steps):
    print(f"✅ Step {index+1} used function `{step._function.name}`")

trace_resultp = True

display(Markdown(f"## ✨ Generated result from the ask: {ask}\n\n---\n" + str(result)))

```

Next we are going to make a sequential planner. Before we made an action planner. 

I want it to do the following. 
Tomorrow is Valentine's day. I need to come up with a poem. Translate the poem to French. This essentially gonna require two 
plugins, essentially one that can write the poem and the other that can translate. 
I'm going to call the planner. 

All right, so we're gonna create plan, async. 
You know, we're built, we're architected in C-sharp where people 
ask, why is there a wait and like async everywhere? 
 
You know, this is enterprise software, people doing stuff. 
So I apologize, but in the end, you will thank us for all of our attendance to things that can happen asynchronously 
because we live in a network world, right? 

Let us print out the plan steps.  So let's see what happens. 
I'm going to bring in the literate friend plugin. 
It's got three functions. 
One is able to write a poem, one is able to translate. I think the other one is to summarize something, 

I'm gonna ask it to make a plan to address this ask and let's see what happens. If things work out the way we want, it's gonna realize that I 
need to write a poem and I need to translate it and so it pulled out two functions to use and you're like well great well can you use them absolutely so what I want to do is see what happens what happens when I have it actually tell me what it created and it 
says tomorrow's the Valentine's. I need to come up with a poem, that's my ask.


```python
from semantic_kernel.planning import SequentialPlanner
from semantic_kernel.core_skills.text_skill import TextSkill
from semantic_kernel.planning.sequential_planner.sequential_planner_config import SequentialPlannerConfig

plugins_directory = "./plugins-sk"
writer_plugin = kernel.import_semantic_skill_from_directory(plugins_directory, "LiterateFriend")

# create an instance of sequential planner, and exclude the TextSkill from the list of functions that it can use.
# (excluding functions that ActionPlanner imports to the kernel instance above - it uses 'this' as skillName)
planner = SequentialPlanner(kernel, SequentialPlannerConfig(excluded_skills=["this"]))

ask = """
Tomorrow is Valentine's day. I need to come up with a poem. Translate the poem to French.
"""
plan = await planner.create_plan_async(goal=ask)
result = await plan.invoke_async()
for index, step in enumerate(plan._steps):
    print(f"✅ Step {index+1} used function `{step._function.name}`")

trace_resultp = True
display(Markdown(f"## ✨ Generated result from the ask: {ask}\n\n---\n" + str(result)))

```

It made a poem and then it translated it to French. Now let's do that in super slow motion.

Let's print out the results step-by-step. So I have a little trace results. 
You can look at the code later, but I'm gonna step through the plan and look at different things inside it and look at the input variables and output variables as they change. 

Add tracing.


```python
from semantic_kernel.planning import SequentialPlanner
from semantic_kernel.core_skills.text_skill import TextSkill
from semantic_kernel.planning.sequential_planner.sequential_planner_config import SequentialPlannerConfig

plugins_directory = "./plugins-sk"
writer_plugin = kernel.import_semantic_skill_from_directory(plugins_directory, "LiterateFriend")

planner = SequentialPlanner(kernel, SequentialPlannerConfig(excluded_skills=["this"]))

ask = """
Tomorrow is Valentine's day. I need to come up with a poem. Translate the poem to French.
"""

plan = await planner.create_plan_async(goal=ask)
planner = SequentialPlanner(kernel, SequentialPlannerConfig(excluded_skills=["this"]))
result = await plan.invoke_async()

for index, step in enumerate(plan._steps):
    print(f"✅ Step {index+1} used function `{step._function.name}`")

trace_resultp = True

if trace_resultp:
    print("Longform trace:\n")
    for index, step in enumerate(plan._steps):
        print("Step:", index)
        print("Description:",step.description)
        print("Function:", step.skill_name + "." + step._function.name)
        print("Input vars:", step._parameters._variables)
        print("Output vars:", step._outputs)
        if len(step._outputs) > 0:
            print( "  Output:\n", str.replace(result[step._outputs[0]],"\n", "\n  "))

display(Markdown(f"## ✨ Generated result from the ask: {ask}\n\n---\n" + str(result)))

```

So I'm gonna run that, and what you'll be able to see is that as 
the kernel takes the plan, the plan has already built a way to take the output from one and stuff it into the input of another, the 
plan has already built a way to take the output from one and stuff it into the input of another. 

Now watch this move here. The poem has been created and the poem's been created and it figured out to add a parameter French. 
So it basically plucked out the fact that I needed to make it in French and it took the poem output and there you have it. 
That is an automatically generated thing. 

Now, you may not think this is a big deal, but it's kind of a big deal because I did not have to tell the system to use those two plugins. 
I just gave it a box of plugins and it just went and pulled out the ones that need it. 
Number two, it created a plan, a multi-step plan to affect a more complex outcome. 

How does this work? There are two dimensions. 
There is completion. The completion is generating the plan. 
Similarity is pulling in context for the completion to be more right. 
It is also pulling out the right plugins through the descriptions to be able to execute a plan. 
Okay, so that was a lot covered in a short amount of time. 
