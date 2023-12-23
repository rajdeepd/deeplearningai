---
layout: default
title: Save the generated content
nav_order: 5
description: ".."
has_children: false
parent:  Coursera - Microsoft Semantic Kernel
---


Think of all the things you're generating, auto-completing with 
this the LLM. 

Let's review inventory, design plugin, This is where we are right now.  What we're going to do is next up 
you did all that completion let's sort of make sure that sparkles here generate 

You can make a kernel with an embedding model. I made a text completion service, I made something called an embedding service. 
I use the text embedding service. 
We already had a service, completion service. 
Well, now you're adding another service. 
It is an embedding service. 
It takes text and converts it to long vectors of numbers. 
You have the completion service and the similarity service. 
Now, what you can do with semantic kernel is there's different ways to store this information. 
What we're gonna do is we're going to draw from the Chroma memory store. 


```python
from IPython.display import display, Markdown
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAITextEmbedding
from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore

kernel = sk.Kernel()

api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_text_completion_service("openai-completion", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id))
kernel.add_text_embedding_generation_service("openai-embedding", OpenAITextEmbedding("text-embedding-ada-002", api_key, org_id))

kernel.register_memory_store(memory_store=ChromaMemoryStore(persist_directory='mymemories'))
print("Made two new services attached to the kernel and made a Chroma memory store that's persistent.")
```
Output generated will be


```
Made two new services attached to the kernel and made a Chroma memory store that's persistent.
```

There's also a volatile memory store. 
There's a Pinecone, Weaviate, Azure Cognitive Search, a bunch of cool ways to kind of like hold on to whatever you generate with embeddings. 
The data that goes into that vector database goes away unless you say persist directory equals and give it a name. 

 
Let's see this works look at that no errors didn't feel good so I did three things there I made a completion service. 
I made a similarity service also called embeddings and I attached a memory store to the kernel so I can hold on to vectors that I generate from 
the embedding conversion process. 
Just a bit of a caveat because we want this notebook to be useful to you. 
Let's say you run this and your memory store which is going to be stored in my 
memories directory starts to give you errors. 
I'm gonna, delete dir equals true. 
If I run this, it just deleted the memory store folder. You might say like, wait, bring it back. 

```python
import shutil

### ONLY DELETE THE DIRECTORY IF YOU WANT TO CLEAR THE MEMORY
### OTHERWISE, SET delete_dir = True

delete_dir = False

if (delete_dir):
    dir_path = "mymemories"
    shutil.rmtree(dir_path)
    kernel.register_memory_store(memory_store=ChromaMemoryStore(persist_directory=dir_path))
    print("⚠️ Memory cleared and reset")
```


Well, first off, let's just go back here and run this code above and no worries, it's there. 
And let's just ignore this code, walk over it, look aside, look 

Okay, next we're gonna do is we're gonna put stuff into the memory store after the embedding vector has been created. 

Okay, so first off, let's get some data. 
You know, I like data, this data from the SWATs. I like the question and answer pairs because those are generally useful as information. 
Right now they're stored in native memory. They haven't gone anywhere. 
Okay, so let's now put all of those strengths, weaknesses, opportunities, and threats into memory. 
So I'm gonna add them to memory collection named SWAT. 
I'm gonna loop over the different arrays of strings and let's just neatly put them all into, there we go. 
So now it's sitting in the vector store. 


```python
strength_questions = ["What unique recipes or ingredients does the pizza shop use?","What are the skills and experience of the staff?","Does the pizza shop have a strong reputation in the local area?","Are there any unique features of the shop or its location that attract customers?", "Does the pizza shop have a strong reputation in the local area?", "Are there any unique features of the shop or its location that attract customers?"]
weakness_questions = ["What are the operational challenges of the pizza shop? (e.g., slow service, high staff turnover)","Are there financial constraints that limit growth or improvements?","Are there any gaps in the product offering?","Are there customer complaints or negative reviews that need to be addressed?"]
opportunities_questions = ["Is there potential for new products or services (e.g., catering, delivery)?","Are there under-served customer segments or market areas?","Can new technologies or systems enhance the business operations?","Are there partnerships or local events that can be leveraged for marketing?"]
threats_questions = ["Who are the major competitors and what are they offering?","Are there potential negative impacts due to changes in the local area (e.g., construction, closure of nearby businesses)?","Are there economic or industry trends that could impact the business negatively (e.g., increased ingredient costs)?","Is there any risk due to changes in regulations or legislation (e.g., health and safety, employment)?"]

strengths = [ "Unique garlic pizza recipe that wins top awards","Owner trained in Sicily at some of the best pizzerias","Strong local reputation","Prime location on university campus" ]
weaknesses = [ "High staff turnover","Floods in the area damaged the seating areas that are in need of repair","Absence of popular calzones from menu","Negative reviews from younger demographic for lack of hip ingredients" ]
opportunities = [ "Untapped catering potential","Growing local tech startup community","Unexplored online presence and order capabilities","Upcoming annual food fair" ]
threats = [ "Competition from cheaper pizza businesses nearby","There's nearby street construction that will impact foot traffic","Rising cost of cheese will increase the cost of pizzas","No immediate local regulatory changes but it's election season" ]

print("✅ SWOT analysis for the pizza shop is resident in native memory")
```
Output generated


```python
✅ SWOT analysis for the pizza shop is resident in native memory
😶‍🌫️ Embeddings for SWOT have been generated
```

So I'm now going to look at this SWAT. The SWAT's all in vector memory, and I'm now going to ask questions of it. 


```python
memoryCollectionName = "SWOT"

for i in range(len(strengths)):
    await kernel.memory.save_information_async(memoryCollectionName, id=f"strength-{i}", text=f"Internal business strength (S in SWOT) that makes customers happy and satisfied Q&A: Q: {strength_questions[i]} A: {strengths[i]}")
for i in range(len(weaknesses)):
    await kernel.memory.save_information_async(memoryCollectionName, id=f"weakness-{i}", text=f"Internal business weakness (W in SWOT) that makes customers unhappy and dissatisfied Q&A: Q: {weakness_questions[i]} A: {weaknesses[i]}")
for i in range(len(opportunities)):
    await kernel.memory.save_information_async(memoryCollectionName, id=f"opportunity-{i}", text=f"External opportunity (O in SWOT) for the business to gain entirely new customers Q&A: Q: {opportunities_questions[i]} A: {opportunities[i]}")
for i in range(len(threats)):
    await kernel.memory.save_information_async(memoryCollectionName, id=f"threat-{i}", text=f"External threat (T in SWOT) to the business that impacts its survival Q&A: Q: {threats_questions[i]} A: {threats[i]}")

print("😶‍🌫️ Embeddings for SWOT have been generated")
```

### Example 1

So, what are the easiest ways to make more money is the question I'm going to ask, and. 
I'm going to do the same kind of memory search async, I'm going to pluck out the different memory results, I'm also going to let you see the relevance score, remember 0 to 1, 1 is like perfect match, 0 is no match. 

```python
potential_question = "What are the easiest ways to make more money?"
counter = 0

memories = await kernel.memory.search_async(memoryCollectionName, potential_question, limit=5, min_relevance_score=0.5)

display(Markdown(f"### ❓ Potential question: {potential_question}"))

for memory in memories:
    if counter == 0:
        related_memory = memory.text
    counter += 1
    print(f"  > 🧲 Similarity result {counter}:\n  >> ID: {memory.id}\n  Text: {memory.text}  Relevance: {memory.relevance}\n")
```
Let's run that. 

Output will be

❓ Potential question: What are the easiest ways to make more money?
  > 🧲 Similarity result 1:
  >> ID: opportunity-0
  Text: External opportunity (O in SWOT) for the business to gain entirely new customers Q&A: Q: Is there potential for new products or services (e.g., catering, delivery)? A: Untapped catering potential  Relevance: 0.7719373733618413

  > 🧲 Similarity result 2:
  >> ID: opportunity-3
  Text: External opportunity (O in SWOT) for the business to gain entirely new customers Q&A: Q: Are there partnerships or local events that can be leveraged for marketing? A: Upcoming annual food fair  Relevance: 0.771400319244832

  > 🧲 Similarity result 3:
  >> ID: opportunity-1
  Text: External opportunity (O in SWOT) for the business to gain entirely new customers Q&A: Q: Are there under-served customer segments or market areas? A: Growing local tech startup community  Relevance: 0.7696970689910045

  > 🧲 Similarity result 4:
  >> ID: opportunity-2
  Text: External opportunity (O in SWOT) for the business to gain entirely new customers Q&A: Q: Can new technologies or systems enhance the business operations? A: Unexplored online presence and order capabilities  Relevance: 0.7673891397036459

  > 🧲 Similarity result 5:
  >> ID: threat-0
  Text: External threat (T in SWOT) to the business that impacts its survival Q&A: Q: Who are the major competitors and what are they offering? A: Competition from cheaper pizza businesses nearby  Relevance: 0.7520414538988753


And so now, it compares what are the easiest ways to make more money to what's in the vector store. 
And this is the first one that's coming up. It's saying catering potential. 
It's saying the annual food fair is coming. And so you see, it's basically sorted the most similar item to the query. 
It's kind of amazing, isn't it? 

### Example 2

Next let's change that. 
Go ahead and change this. It's kind of like an amazing feeling. 
What are the easiest ways to save money? 

Let's see what it does with that one. 
It says partnerships. It says, worry about your competition. 
The cheese, don't forget the cheese. And so again, this is a magical machine now that takes your gravy drippings and uses it. 
And this kind of, remember left hand, right hand? 
This is your left hand doing amazing things. Okay, so let's go into a super long example. 


Now, I think you're kind of tired of that long example. 
So let me give you something a little easier because typing this is kind of hard. 

Okay, let's read this code here for a second. All right, let's have a what if scenario. 
**how can the business owner save time?**


```python
what_if_scenario = "How can the business owner save time?"
counter = 0

gathered_context = []
max_memories = 3
memories = await kernel.memory.search_async(memoryCollectionName, what_if_scenario, limit=max_memories, min_relevance_score=0.77)

print(f"✨ Leveraging information available to address '{what_if_scenario}'...")

for memory in memories:
    if counter == 0:
        related_memory = memory.text
    counter += 1
    gathered_context.append(memory.text + "\n")
    print(f"  > 🧲 Hit {counter}: {memory.id} ")

skillsDirectory = "./plugins-sk"
print(f"✨ Synthesizing human-readable business-style presentation...")
pluginFC = kernel.import_semantic_skill_from_directory(skillsDirectory, "FriendlyConsultant");

my_context = kernel.create_new_context()
my_context['input'] = what_if_scenario
my_context['context'] = "\n".join(gathered_context)

preso_result = await kernel.run_async(pluginFC["Presentation"], input_context=my_context)

display(Markdown("# ✨ Generated presentation ...\n"+str(preso_result)))

```

It's going to do the memory search to find the most similar memories. 
I'm gonna use a plugin, a plugin from the friendly consultant folder, plugin collection. 
I'm gonna ask it to give me a presentation. I've made a plugin to make a presentation about anything I ask it to do. 
And long story short, set the context. And I ask it to run. 


```python
what_if_scenario = "How can the business owner save time?"
counter = 0

gathered_context = []
max_memories = 3
memories = await kernel.memory.search_async(memoryCollectionName, what_if_scenario, limit=max_memories, min_relevance_score=0.77)

print(f"✨ Leveraging information available to address '{what_if_scenario}'...")

for memory in memories:
    if counter == 0:
        related_memory = memory.text
    counter += 1
    gathered_context.append(memory.text + "\n")
    print(f"  > 🧲 Hit {counter}: {memory.id} ")

skillsDirectory = "./plugins-sk"
print(f"✨ Synthesizing human-readable business-style presentation...")
pluginFC = kernel.import_semantic_skill_from_directory(skillsDirectory, "FriendlyConsultant");

my_context = kernel.create_new_context()
my_context['input'] = what_if_scenario
my_context['context'] = "\n".join(gathered_context)

preso_result = await kernel.run_async(pluginFC["Presentation"], input_context=my_context)

display(Markdown("# ✨ Generated presentation ...\n"+str(preso_result)))
```
✨ Leveraging information available to address 'How can the business owner save time?'...
  > 🧲 Hit 1: opportunity-2 
  > 🧲 Hit 2: opportunity-0 
  > 🧲 Hit 3: opportunity-1 
✨ Synthesizing human-readable business-style presentation...

```
Business Strategy Consultant Presentation
Summary
The business owner has asked for ways to save time. We will explore three key concerns that can help the business owner save time and improve operations.

The Question
How can the business owner save time?

Three Key Concerns
Streamlining online presence and order capabilities
Tapping into catering potential
Targeting the growing local tech startup community
Streamlining Online Presence and Order Capabilities
The business can save time by exploring new technologies or systems to enhance operations. By improving the online presence and order capabilities, the business can attract entirely new customers. For example, implementing an online ordering system can save time by reducing the need for phone orders and manual data entry.

Tapping into Catering Potential
The business can save time by exploring new products or services, such as catering. By tapping into the untapped catering potential, the business can attract entirely new customers and increase revenue. For example, offering catering services can save time by streamlining the ordering process and reducing the need for individual orders.

Targeting the Growing Local Tech Startup Community
The business can save time by targeting under-served customer segments or market areas. By targeting the growing local tech startup community, the business can attract entirely new customers and increase revenue. For example, offering discounts or promotions to local tech startups can save time by reducing the need for individual marketing efforts.

Summary
By streamlining online presence and order capabilities, tapping into catering potential, and targeting the growing local tech startup community, the business owner can save time and improve operations.


```


Let's see how this works. 
So first off, it's used a similarity engine 
to find the most similar pieces of context. 
It's going to take all of that and give it to the prompt that is going to generate the presentation. 
So this is that example of retrieval augmented generation. 
The generated information is taken from the actual information stored in the vector database. 

This is a professional presentation from a consultant. The question is, how can the business owner save time? 

Here are the three concerns. 
Here's how to address them individually. And this is what I brought to you. 
 
Remember, you can change everything in these notebooks, 
like not just this one with other ones, and you can do entirely different 
analyses. 
I want to congratulate you because you've now unlocked this very 
popular acronym called RAG. and you've accessed both the completion and similarity engines. 
Congratulations. 
You were able to bind similar things in the vector database. You were able to give them to a completion prompt and generate something on point. 

And now that you've done this, we're going to take you into the next chapter, which is all about something at a whole different level. 
Once you master plugins and similarity completion, all this world, world, you suddenly discover that it's time to make a plan, make an AI plan, generate a way to solve a goal instead of just sift through your plugins by hand. 
Have the AI go look through your giant Lego made of plugins. 
You don't have to go look through them. The AI can look through them. 

Let's jump into plans. 
