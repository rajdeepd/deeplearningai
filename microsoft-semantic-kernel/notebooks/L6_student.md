# 🧑‍🍳 L6 - A kitchen that responds to your “I’m hungry” is more than feasible

Inventory:

1. Kernel
2. Semantic (and Native) functions -- you can do a lot with these
3. BusinessThinking plugin --> SWOTs in ways you could never imagine
4. DesignThinking plugin --> you did that. Congrats
5. Use the similarity engine to your heart's content 🧲
6. THE BIG ONE!!!!!

# 🔥 Let's make a kernel one more time!


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

We want to have a vat of plugins ... and then find the right plugin to fit the goal ...

**Note**: You can find more about the predefined plugins used below [here](https://learn.microsoft.com/en-us/semantic-kernel/ai-orchestration/out-of-the-box-plugins?tabs=Csharp).


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


```python
ask = "What is the sum of 110 and 990?"

print(f"🧲 Finding the most similar function available to get that done...")
plan = await planner.create_plan_async(goal=ask)
print(f"🧲 The best single function to use is `{plan._skill_name}.{plan._function.name}`")

```


```python
ask = "What is today?"
print(f"🧲 Finding the most similar function available to get that done...")
plan = await planner.create_plan_async(goal=ask)
print(f"🧲 The best single function to use is `{plan._skill_name}.{plan._function.name}`")


```


```python
ask = "How do I write the word 'text' to a file?"
print(f"🧲 Finding the most similar function available to get that done...")
plan = await planner.create_plan_async(goal=ask)
print(f"🧲 The best single function to use is `{plan._skill_name}.{plan._function.name}`")


```

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

# 🔖 There are a variety of limitations to using the planner in August of 2023 in terms of number of tokens required and model preference that we can expect to slowly vanish over time. For simple tasks, this Planner-based approach is unusually powerful. It takes full advantage of both COMPLETION and SIMILARITY in a truly magical way.

![](./assets/twodimensions.png)


```python

```


```python

```
