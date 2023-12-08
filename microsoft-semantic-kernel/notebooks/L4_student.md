# 🧑‍🍳 L4 - Frozen dinner: The design thinking meal

Inventory:

1. Kernel
2. Semantic (and Native) functions -- you can do a lot with these
3. BusinessThinking plugin --> SWOTs in ways you could never imagine
4. DesignThinking plugin ... Here you are

# 🔥 Get a kernel ready


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

    A kernel is now ready.


## 🏁 Let's start backwards from the customer

```directory
plugins-sk/
│
└─── DesignThinking/
     |
     └─── Define/
     |    └─── config.json
     |    └─── skprompt.txt
     |
     └─── Empathize/
          └─── config.json
          └─── skprompt.txt

```


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


## ✨ The categorized observations from the 'Empathize' phase of design thinking



    [
      {
        "sentiment": "Negative",
        "summary": "Complaints about the condition of the seats and the cleanliness of the restaurant."
      },
      {
        "sentiment": "Positive",
        "summary": "Praise for the garlic pizza and its unique taste."
      },
      {
        "sentiment": "Negative",
        "summary": "Frustration with the high turnover rate of servers and their lack of knowledge."
      },
      {
        "sentiment": "Neutral",
        "summary": "Questioning the absence of calzones on the menu."
      },
      {
        "sentiment": "Negative",
        "summary": "Displeasure with the high prices of the pizza and the lack of online ordering options."
      }
    ]


![](./assets/designthinking.png)


**Note:** LLMs do not necessarily produce the same result each time. Your results may be different than the video.


```python
my_result = await kernel.run_async(pluginDT["Empathize"], pluginDT["Define"], input_str = customer_comments)

display(Markdown("## ✨ The categorized observations from the 'Empathize' + 'Define' phases of design thinking\n"+str(my_result)))
```


## ✨ The categorized observations from the 'Empathize' + 'Define' phases of design thinking
| Analysis | Definition | Possible Source |
| --- | --- | --- |
| Complaints about seating and server consistency | Customers express dissatisfaction with the condition of the seats and the lack of consistency with servers. | Poor maintenance of seating and inadequate training of servers. |
| Praise for garlic pizza | Customers express admiration for the garlic pizza, which is a unique and exceptional dish. | High quality ingredients and skilled preparation of the dish. |
| Questions about absence of calzones and high pizza prices | Customers ask about the absence of calzones and express concern about the high price of the pizza. | Limited menu options and high cost of ingredients. |
| Frustration with lack of online ordering and dirty seating | Customers express frustration with the lack of online ordering and the uncomfortable and dirty seating. | Inadequate investment in technology and poor maintenance of seating. |
| Enthusiasm for garlic pizza | Customers express excitement for the garlic pizza, which is a beloved and unmatched menu item. | High quality ingredients and skilled preparation of the dish. |



```python
my_result = await kernel.run_async(pluginDT["Empathize"], pluginDT["Define"], pluginDT["Ideate"], pluginDT["PrototypeWithPaper"], input_str=customer_comments)

display(Markdown("## ✨ The categorized observations from the 'Empathize' + 'Define' + 'Ideate' + 'Prototype' + phases of design thinking\n"+str(my_result)))
```


## ✨ The categorized observations from the 'Empathize' + 'Define' + 'Ideate' + 'Prototype' + phases of design thinking
1. Regular cleaning and maintenance of seating areas: Create a cleaning schedule with specific tasks and assign responsibilities to employees. Use a paper checklist to track completion of tasks and note any issues that need to be addressed.

2. Offer discounts for customers who bring their own seating cushions: Create a paper coupon that can be handed out to customers who bring their own cushions. The coupon should clearly state the discount amount and any restrictions.

3. Introduce a loyalty program for repeat customers: Create a paper loyalty card that can be stamped or punched each time a customer makes a purchase. The card should have clear instructions on how to earn rewards and what those rewards are.

4. Train servers to be knowledgeable about menu offerings: Create a paper training manual that includes detailed information about each menu item, including ingredients, preparation methods, and suggested pairings. Use quizzes or role-playing exercises to test server knowledge.

5. Offer free garlic bread with every pizza order: Create a paper coupon that can be handed out to customers with their pizza order. The coupon should clearly state the offer and any restrictions.

6. Add calzones to the menu: Create a paper menu insert that includes the new calzone options. The insert should be visually appealing and clearly labeled as "New Menu Items."

7. Implement a feedback system for customers to share their experiences: Create a paper feedback form that can be handed out to customers with their bill. The form should include questions about the customer's experience, as well as space for comments and suggestions.

8. Offer a gluten-free pizza option: Create a paper menu insert that includes the new gluten-free pizza option. The insert should be visually appealing and clearly labeled as "Gluten-Free Options."

9. Introduce a delivery service: Create a paper delivery menu that includes all menu items available for delivery, as well as delivery fees and minimum order amounts. The menu should be visually appealing and easy to read.

10. Host events and promotions to attract new customers: Create a paper flyer or poster advertising the event or promotion. The flyer should include all relevant details, such as date, time, location, and any special offers or discounts.



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


### ✨ As an AI language model, I cannot predict the exact reactions of the 40-year old man in this scenario. However, some possible reactions could be:

- They might feel indifferent and not care about the loyalty card.
- They might feel annoyed or frustrated that they have to keep track of points and rewards.
- They might feel intrigued and motivated to earn rewards by visiting the bar more often.
- They might feel grateful and appreciated for being offered a loyalty program.
- They might feel confused or overwhelmed by the details of the loyalty card and need clarification.


## 🔖 Reminder: We haven't explicitly used the 🧲 similarity engine — we'll be doing that next! 

![](./assets/twodimensions.png)


```python

```
