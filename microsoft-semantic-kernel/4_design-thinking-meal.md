---
layout: default
title: Design thinking Meal
nav_order: 4
description: ".."
has_children: false
parent:  Coursera - Microsoft Semantic Kernel
---


Let's build a design thinking plugin. 
 
Design thinking is the five-step process, empathize, define, ideate, prototype, test. 

<img src="/deeplearningai/microsoft-semantic-kernel/images/Screenshot_2023-11-22_at_8.20.01 PM.png" width="70%"/>
<img src="/deeplearningai/microsoft-semantic-kernel/images/Screenshot_2023-11-22_at_8.20.12 PM.png" width="70%"/>

. 
The Design Thinking plugin is instrumental in streamlining the design thinking process. Its significance lies in its ability to transform AI capabilities into modular, transportable plugins. These plugins can be conveniently packaged for use by others, facilitating their application in various contexts. This approach serves as an effective method for disseminating knowledge and techniques, akin to sharing recipes.

Now, let us conduct a brief review of our current capabilities:

Proficiency in creating a kernel.
Ability to develop both semantic and native functions.
Competence in utilizing business thinking tools, logging in, and processing SWOT analyses in previously unimagined ways.
While it's uncertain whether these skills were envisaged prior to this course, their impressive nature is undeniable. Our next step involves engaging with the Design Thinking plugin more deeply.

To commence, our priority is to prepare a kernel.


```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from IPython.display import display, Markdown

kernel = sk.Kernel()

useAzureOpenAI = False

if useAzureOpenAI:
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    kernel.add_text_completion_service("azureopenai", AzureChatCompletion(deployment, endpoint, api_key))
else:
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_text_completion_service("openai", OpenAIChatCompletion("gpt-3.5-turbo-0301", api_key, org_id))

print("A kernel is now ready.")    
```

## Let's start backwards from the customer

Next, our focus will shift to a critical aspect of our process: initiating design thinking based on customer feedback. To begin, we will revisit the customer's perspective.

Initially, it is essential to reintegrate various elements into our analysis, specifically the slot questions and their corresponding responses, due to their relevance and utility. Subsequently, the next step involves the collection of a comprehensive array of customer feedback.

Let's get some customer comments. 

```python
import json

pluginsDirectory = "./plugins-sk"

strength_questions = ["What unique recipes or ingredients does the pizza shop use?","What are the skills and experience of the staff?","Does the pizza shop have a strong reputation in the local area?","Are there any unique features of the shop or its location that attract customers?", "Does the pizza shop have a strong reputation in the local area?", "Are there any unique features of the shop or its location that attract customers?"]
weakness_questions = ["What are the operational challenges of the pizza shop? (e.g., slow service, high staff turnover)","Are there financial constraints that limit growth or improvements?","Are there any gaps in the product offering?","Are there customer complaints or negative reviews that need to be addressed?"]
opportunities_questions = ["Is there potential for new products or services (e.g., catering, delivery)?","Are there under-served customer segments or market areas?","Can new technologies or systems enhance the business operations?","Are there partnerships or local events that can be leveraged for marketing?"]
threats_questions = ["Who are the major competitors and what are they offering?","Are there potential negative impacts due to changes in the local area (e.g., construction, closure of nearby businesses)?","Are there economic or industry trends that could impact the business negatively (e.g., increased ingredient costs)?","Is there any risk due to changes in regulations or legislation (e.g., health and safety, employment)?"]

strengths = [ "Unique garlic pizza recipe that wins top awards","Owner trained in Sicily","Strong local reputation","Prime location on university campus" ]
weaknesses = [ "High staff turnover","Floods in the area damaged the seating areas that are in need of repair","Absence of popular calzones from menu","Negative reviews from younger demographic for lack of hip ingredients" ]
opportunities = [ "Untapped catering potential","Growing local tech startup community","Unexplored online presence and order capabilities","Upcoming annual food fair" ]
threats = [ "Competition from cheaper pizza businesses nearby","There's nearby street construction that will impact foot traffic","Rising cost of cheese will increase the cost of pizzas","No immediate local regulatory changes but it's election season" ]

```
Presented herein is an overview of our business, structured in a question-and-answer format. Included are customer observations, comprising ten remarks that vary between positive and negative sentiments. The objective now is to access the Design Thinking plugin, which, as you are aware, resides within a specific folder.

<img src="/deeplearningai/microsoft-semantic-kernel/images/Screenshot_2023-11-30_at_12.19.31 PM.png" width="100%"/>

It is prudent to orient ourselves in this context. This folder, which I shall demonstrate, houses the Design Thinking plugin along with other components such as the `Plugins SK` and the `Define and Empathize SK prompt.txt`. To visualize these elements, I encourage you to initiate your file browser. Should you wish to examine the 'Empathize' prompt, it can be found within this setup, appearing in a manner as described here.

```
The following are anonymized comments from customers:
---
{{$input}}
---
Isolate the five most common sentiments from 
customer feedback as a response in JSON format. 
The format should read:
[
    { "sentiment": "expression of a sentiment", 
        "summary": "concise summary of reason for this sentiment"},
    { "sentiment": "expression of a sentiment", 
        "summary": "concise summary of reason for this sentiment"},
    { "sentiment": "expression of a sentiment", 
        "summary": "concise summary of reason for this sentiment"},
    { "sentiment": "expression of a sentiment", 
        "summary": "concise summary of reason for this sentiment"},
    { "sentiment": "expression of a sentiment", 
        "summary": "concise summary of reason for this sentiment"},
]
```

The document outlines the utilization of various plugins and design thinking methodologies, specifically focusing on the 'Empathize' plugin, which is detailed within the `SK prompt.txt` file. To review these elements, one should initiate their file browser. The `Empathize` plugin is particularly adept at analyzing anonymized customer comments, transforming these inputs into a JSON format that lists sentiments. This process leverages the plugin's proficient sentiment analysis capabilities.

Further, the document explains that while customer comments will be utilized by the `Empathize` plugin, other elements such as SWOTs (Strengths, Weaknesses, Opportunities, Threats) will be reserved for later use. The procedure involves activating the `Design Thinking` plugin within the plugin directory. This includes importing the plugin and executing the kernel using the 'empathize' function.

<img src="/deeplearningai/microsoft-semantic-kernel/images/Screenshot_2023-11-30_at_12.49.12 PM.png" width="100%"/>

The process then involves taking the customer comments and conducting an empathy analysis using the data from the `SK prompt.txt` file located in the specified directory. It is important to note that any modifications made to this file and subsequently saved will result in changes to the prompt used for analysis.

<img src="/deeplearningai/microsoft-semantic-kernel/images/design-thinking-image1.png"  width="120%"/>


```python
customer_comments = """
Customer 1: The seats look really raggedy.
Customer 2: The garlic pizza is the best on this earth.
Customer 3: I've noticed that there's a new server every time I visit, and they're clueless.
Customer 4: Why aren't there calzones?
Customer 5: I love the garlic pizza and can't get it anywhere else.
Customer 6: The garlic pizza is exceptional.
Customer 7: I prefer a calzone's portable nature as compared with pizza.
Customer 8: Why is the pizza so expensive?
Customer 9: There's no way to do online ordering.
Customer 10: Why is the seating so uncomfortable and dirty?
"""

pluginDT = kernel.import_semantic_skill_from_directory(pluginsDirectory, "DesignThinking");
my_result = await kernel.run_async(pluginDT["Empathize"], input_str=customer_comments)

display(Markdown("## ✨ The categorized observations from the 'Empathize' phase of design thinking\n"))

print(json.dumps(json.loads(str(my_result)), indent=2))
```






<img src="/deeplearningai/microsoft-semantic-kernel/images/Screenshot_2023-11-30_at_12.49.27 PM.png" width="100%"/>
<img src="/deeplearningai/microsoft-semantic-kernel/images/Screenshot_2023-11-30_at_12.49.35 PM.png" width="100%"/>

<img src="/deeplearningai/microsoft-semantic-kernel/images/Screenshot_2023-11-30_at_12.32.38 PM.png" width="100%"/>

Do it yourself. 
And you should do it. Let me show you it 
running so we can look at the food and pick 
it out around on our dish. 
So what did it do? 
It categorized different types of complaints 
about seat condition, praise for the garlic pizza. 
Someone likes it. 
Frustration because always a new person is serving them, doesn't 
know the layout of the restaurant. 
It's like a neutral response, like why aren't there calzones? 
And lastly, there's no online ordering. 
The pizza shop owner hasn't done any digital transformation yet. 
So what do we have here? 
We have a design thinking, instant empathize, magical AI, 
generative moment that took in these comments and it generated a sentiment map. 

Design thinking is a simple set of steps. 
There are five steps. 

Most people argue that there's no steps in design thinking, that 
it's a set of activities you can sort of do in any order, really. 
But we did empathize, and we used large language model 
AI to summarize the feedback. 
Now once you have this feedback, you can then convert it into input to defining the problem. You know, you can never solve a problem unless you understand it well, as you define it.

So let's rerun the plugin with the original information. 
It's calculating that. Okay. These are the responses. And now I'm going to send it into the define plugin. Here we go. 


```python
my_result = await kernel.run_async(pluginDT["Empathize"], pluginDT["Define"], input_str = customer_comments)

display(Markdown("## ✨ The categorized observations from the 'Empathize' + 'Define' phases of design thinking\n"+str(my_result)))
```
<img src="/deeplearningai/microsoft-semantic-kernel/images/Screenshot_2023-11-30_at_12.32.38 PM.png" width="100%"/>


```python
my_result = await kernel.run_async(pluginDT["Empathize"], pluginDT["Define"], pluginDT["Ideate"], pluginDT["PrototypeWithPaper"], input_str=customer_comments)

display(Markdown("## ✨ The categorized observations from the 'Empathize' + 'Define' + 'Ideate' + 'Prototype' + phases of design thinking\n"+str(my_result)))
```


I'm going to use the empathize plugin. 
I'm going to feed that output into plugin define.
 
And then I am going to give as an input string, I'm going to give the feedback, customer comments, And then I am going to output them. Let's get a fancy plated statement there. 


```python
sk_prompt = """
A 40-year old man who has just finished his shift at work and comes into the bar. They are in a bad mood.

They are given an experience like:
{{$input}}

Summarize their possible reactions to this experience.
"""
test_function = kernel.create_semantic_function(prompt_template=sk_prompt,
                                                    description="Simulates reaction to an experience.",
                                                    max_tokens=1000,
                                                    temperature=0.1,
                                                    top_p=0.5)
sk_input="""
A simple loyalty card that includes details such as the rewards for each level of loyalty, how to earn points, and how to redeem rewards is given to every person visiting the bar.
"""

test_result = await kernel.run_async(test_function, input_str=sk_input) 

display(Markdown("### ✨ " + str(test_result)))
```

 
