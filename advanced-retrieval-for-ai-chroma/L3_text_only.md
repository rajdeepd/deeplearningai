Field of information retrieval has been around for a while as a subfield of natural language processing, and there's many approaches to improving the relevancy of query results. But what's new is we have powerful large language models, and we can use those to augment and enhance the queries that we send to our vector-based retrieval system to get better results. Let's take a look.

## Expansion with Generated answers

So the first type of query expansion we're going to talk about is called expansion with generated answers.
Typically the way that this works is you take your query and you pass it over to an LLM which you prompt to generate a hypothetical or imagined answer to your query and then you concatenate your query with the imagined answer and use that as the new query which you pass to your retrieval system or vector database.
Then you return your query results as normal. Let's take a look at how this works in practice.

So the first thing that we're going to do is once again and grab all the utilities that we need.
We're going to load everything we need from Chroma and create our embedding function.
We're going to set up our OpenAl client again, because we'll be using the LLM.
And once again, to help with visualization, we're going to use UMAP and project our data set so that that's all ready to go for us.
Now that we're done setting up, let's take a look at expansion with generated answers and there's a reference here to the paper which demonstrates some of the empirical results that you can get by applying this method.
So to do expansion with generated answers we're yoing to use an LLM, in this case GPT, and just the same as last time we're going to prompt the model in a particular way.
Let's create this function called augment query generated and we're going to pass in a query, we're also going to pass in a model argument, in this case GPT 3.5 turbo by default, and we're going to prompt the model and in the system prompt we're going to say you're a helpful expert financial research assistant, provide an example answer to the given question that might be found in a document, like an annual report.
In other words, we're pretty much asking the model to hallucinate, but we're gonna use that hallucination for something useful.
And in the user prompt, we're just gonna pass the query as the content.
And then we'll do our usual unpacking of the response.
And that defines how we're going to prompt our model.
Let's wire this together.
Here's our original query, asking was there a significant turnover in the executive team.
We will generate a hypothetical answer and then we'll create our joint query which is basically the original query prepending the hypothetical answer.
Now let's take a look at what this actually looks
like after we generate it.
So here we see the output. We see our original query, was there a significant turnover in the executive team, and a hypothetical answer.
In the past fiscal year there was no
significant turnover in the executive team, the core members of the executive team remained unchanged, etc.
So let's send this query plus the hypothetical critical response to our retrieval system as a query.
And we'll querv the Chroma Collection the usual way and print out our results.
And we're sending the joint query as the query to our retrieval system. And we're retrieving the documents and the embeddings together. So these are the documents that we get. Back, we see things here discussing leadership. We see how consultants and directors work together. Here we have an overview of the different directors that we had in Microsoft. And we talk about the different board committees. Let's visualize this. Let's see what sort of difference this made. So, to do that, we get our retrieved embeddings, we get the embedding for our original query, we get the embedding for our joint query, and then we project all three. I'm plotting the projection. And we see the red $X$ is our original query, the orange $X$ is our new query with the hypothetical answer. And we see that we get this nice cluster of results, but most importantly, what I want to illustrate here is that using the hypothetical answer moves our query elsewhere in space, hopefully producing better results for us.
So that was query expansion with generated queries, but there's another type of query expansion we can also try. This is called query expansion with multiple queries. And the way that you use this is to use the LLM to generate additional queries that might help answering the question. So what you do here is you take your original query, you pass it to the LLM, you ask the. LLM to generate several new related queries to the same original query, and then you pass those new queries along with your original query to the vector database or your retrieval system. That gives you results for the original

and the new queries, and then you pass all of those results to the LLM to complete the RAG loop.
So let's take a look at how this works in practice.
Once again, the starting point is a prompt in the model, and we see here that we have a system prompt, and the system prompt is a bit more detailed this time. We take in a query, which is our original query, and we ask the model. It's a helpful expert financial research assistant. The users are asking questions about an annual report, so this gives the model enough context to know what sorts of queries to generate. And then you say, suggest up to five additional related questions to help them find the information they need for the provided question. Suggest only short questions without compound sentences, and this makes sure that we get simple queries.
Suggest a variety of questions that cover different aspects of the topic, and this is very important because there are many ways to rephrase the same query, but what we're actually asking for is different but related queries.
And finally, we want to make sure that they're complete questions, they're related to the original question, and we ask some formatting output.
One important thing to understand about these techniques in particular that bring an LLM into the loop of retrieval is prompt engineering becomes a concern. It's something that you have to think about and I really recommend that you as a student play with these prompts once you have a chance to try the lab.
See how they may change, see what different types of queries you can get the models to generate,

and the new queries, and then you pass all of those results to the LLM to complete the RAG loop.
So let's take a look at how this works in practice.
Once again, the starting point is a prompt in the model, and we see here that we have a system prompt, and the system prompt is a bit more detailed this time. We take in a query, which is our original query, and we ask the model. It's a helpful expert financial research assistant. The users are asking questions about an annual report, so this gives the model enough context to know what sorts of queries to generate. And then you say, suggest up to five additional related questions to help them find the information they need for the provided question. Suggest only short questions without compound sentences, and this makes sure that we get simple queries.
Suggest a variety of questions that cover different aspects of the topic, and this is very important because there are many ways to rephrase the same query, but what we're actually asking for is different but related queries.
And finally, we want to make sure that they're complete questions, they're related to the original question, and we ask some formatting output.
One important thing to understand about these techniques in particular that bring an LLM into the loop of retrieval is prompt engineering becomes a concern. It's something that you have to think about and I really recommend that you as a student play with these prompts once you have a chance to try the lab.
See how they may change, see what different types of queries you can get the models to generate,

array of augmented queries. So now we have one array where each entry is a query, our original query, plus the augmented queries.
And we can grab the results.
And again, Chroma can do querying in batches.
And let's look at the retrieved documents that we get.
And one thing that's important here is because the queries are related, you might get the same document retrieved for more than one query. So what we need to do is to deduplicate the retrieved and that's what we do here. And finally, let's just output the documents that we get. So we can see now the documents that we get for each query.
And these are all to do with revenue, different aspects of revenue growth, which is exactly what we were hoping for.
We have increases in Windows revenue.
We can see things that are coming from other components.
So for example, what were the most important factors that contributed to decreases in revenue? So we see increased sales and marketing expenses, different types of investments, different types of tax breaks.
Essentially, each of these augmented queries are providing us with a slightly different set of results. And let's visualize that. What did we actually get in geometric space in response to these results?
So again, we'll take our original query embedding and our augmented query embeddings and project them.
And the next thing we'll do is project our result embeddings.
Before we do that, we need to flatten the list because we have a list of embeddings per query.
We just want a flat list of returned embeddings.
And then we just project them as before.
And then let's visualize what we get.
And we see that using query expansion, we were able to actually hit other related parts of the data set that our single original query may not have reached. And this gives us more of a chance to find all of the related information, especially in the context of more complex queries, which require more and different types of information to answer. So here we see that the red $X$ is our original query, the orange X's are the augmented, the new queries generated for us by the LLM.
And then once again, the green circles represent the results that we actually return by the retrieval system to the model. One way to think about this is that a single query turns into a single point in embedding space. And a single point in embedding space likely doesn't contain all of the information that you need to answer a more complex query like this one. So using this form of query expansion, where we generate multiple related queries using an LLM, gives us a better chance of capturing all of the related information.
The downside of this, of course, is now we have a lot more results than we had originally. And we're not sure if and which of these results are actually relevant to our query. In the next lab, using cross-encoder re-ranking, we have a technique that allows us to actually score the relevancy of all the returned results and use only the ones we feel match our original query. And l'll demonstrate that in the next lab. In this lab, I ecommend that you try playing around with the query expansion prompts, ry your own queries, and see the types of results you

get by asking different types of questions about the Microsoft Annual Report.