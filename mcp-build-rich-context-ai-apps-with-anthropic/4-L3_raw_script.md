0:02 Now it's time to start coding.
0:04 In this lesson, you'll build a chatbot and code its tools.
0:07 Before you start working with MCP, let's make sure you have a good foundation
0:11 with tools use and prompting large language models.
0:14 All right, let's code.
0:15 So let's get started building our chatbot.
0:17 And before we hop into building MCP servers, we're going to start
0:21 just by building a very simple application with a chatbot that makes
0:25 use of archive for searching for papers and finding some information.
0:29 If this is information that you're already familiar with, feel
0:32 free to skip this lesson and hop into building your first MCP server.
0:35 So let's get started by bringing in some libraries that we need.
0:38 So we're going to bring in the arxiv SDK.
0:40 This is going to allow us to start searching for papers.
0:43 We're then going to bring in the JSON
0:44 module for some formatting the OS module for environment variables.
0:48 We're going to bring in a little bit of typing so we can type our code.
0:51 And then we'll make sure we bring in the Anthropic SDK.
0:54 I'm going to first start
$\underline{0: 55}$ by defining a constant here called paper directory,
$0: 58$ which is just going to be the string papers.
1:00 And this is what I'm going to be using for saving information to the file system.
1:04 I'm then going to bring in my first function here.
1:06 So let's go take a look at this.
1:07 This function is called search papers.
1:08 And it accepts some kind of topic and a number of results which defaults to five.
1:12 And what I'm doing here is searching for papers on archive.
1:15 And if you're not familiar with Archive is an open source repository
1:19 of many different published papers across a variety of domains,
1:22 from mathematics to science to many different disciplines.
1:26 So we're going to search for some papers,
1:27 and then we're going to return a list of paper IDs.
1:30 And we're going to then use those paper IDs
1:32 in another function to get more detail and summarization.
1:35 We're going to initialize our client.
1:36 And then we'll start searching for relevant articles.
1:39 We'll take the results of those search and we'll go ahead and create a directory
1:42 if it exists already, great.
1:43 If not, we'll go ahead and make it.
1:45 And we'll save this information to a file called papers info dot JSON.
1:49 What we're going to do is process each of these papers and create a dictionary.
1:52 And then we're going to go ahead and write that to our file.
1:55 This is going to give us back some IDs when we're done.
1:57 Let's go ahead and search for some papers for computers.
2:00 We'll see here. This is being saved to a file locally.

2:00 We'll see here. This is being saved to a file locally.
2:02 And we got a bunch of IDs that we can use to get some more information around.
2:06 We're going to go ahead and bring in our second function here
2:08 now to make use of this paper ID.
2:10 So we're going to define another function here called extract info.
$\underline{2: 13}$ Which is going to take in one of these paper IDs
$\underline{2: 16}$ it's going to look in our papers info JSON and give us back some information
$\underline{2: 19}$ about that paper.
2:21 And if it can't be found we'll go ahead and just return a string.
$\underline{2: 23}$ There's no saved information for that paper.
$\underline{2: 26}$ So I'm going to go ahead and grab this first ID right here.
2:29 Let's go ahead and run this function.
2:30 And then we'll go ahead and call this function
$\underline{2: 33}$ with that particular ID just to show you what this looks like.
2:35 We can see right here we're getting back some data
2:38 not only related to the title of this, but also the URL
2:41 for the PDF as well as a summary of this particular paper.
2:44 We're going to start with these two functions.
2:46 And then we're going to go ahead and start
2:47 bringing in these functions as tools for a large language model.
2:51 So what we're going to be able to do is pass in some tools for Anthropic's
$\underline{2: 55}$ Claude model.
2:56 We're then going to go ahead and build a small chatbot that takes in these tools
3:00 and knows when to call them and return the data, particularly for these functions.
3:04 So let's define our tools list.

3:06 If you are familiar with tool use, this should be nothing terribly new,
3:09 but every single tool that you make is always
3:11 going to have a name and a description
3:13 and then some kind of schema that it needs to follow.
3:15 So in this case, we have a tool for search papers and a tool for extract info.
3:19 Remember that the model itself is not going to call these functions.
3:23 We actually need to write the code
3:24 to call those functions and pass the data back to the model.
3:28 But these tools are going to allow the model to extend its functionality.
3:31 So instead of saying I don't know or hallucinate,
3:34 we're going to get back the answer that we want here.
3:36 So let's go ahead and start writing some of our logic
3:38 for working with our large language model and executing these tools.
3:42 Let's bring in a little bit of mapping for our tool.
3:45 And right here we've got a function that is going to map the tools
3:47 that we have to calling that underlying function.
3:51 What you can see here is we have a dictionary for each of our tool names.
3:54 These refer to the functions that we have below.
3:56 And then a handy helper function to then go ahead and call that particular function
4:00 and return the result to us in a variety of data types that come in.
4:03 Let's go ahead and start building in our chatbot right now.

4:06 That's going to start by bringing in any environment variables that we have
4:08 API keys, and then creating an instance of our Anthropic client.
4:13 We're going to need this
$\underline{4: 13}$ so that we can go ahead and make calls to our model and get back some data.
4:17 Let's go ahead and bring in our boilerplate function
4:20 to go ahead and start working with the model.
4:22 If you've worked with Anthropic
4:23 before or many other models, this is going to look very familiar.
4:27 We're going to start with a list of messages.
4:29 We're going to go ahead and pass in the query that the user puts in.
4:32 I'm going to talk to Claude 3.7 Sonnet right now.
4:34 We're going to go ahead and start a loop for the chat.
4:37 And if there is text data put that into the message.
$\underline{4: 40}$ If the data that's coming in is tool use, if the model detects
$\underline{4: 43}$ that a tool needs to be used,
4:44 we're going to go ahead and bring in our helper function for executing the tool
$\underline{4: 48}$ and then appending that tool result to the list of messages.
4:51 Let's go ahead and see this in action. Bring in our chat loop.
4:54 We've got all the functionality
$\underline{4: 55}$ we need to start working with tool use talking to our model.
4:58 Now let's start with a very simple example
5:00 for what it's going to look like to actually use this function.
5:03 We're going to run an infinite loop until we pass in the string quit.
5:06 So let's go ahead and start talking to our large language model.
5:09 Call our chat loop function.

5:11 Right now we can start putting in a query to start talking to our model.
5:14 So let's start with something very simple,
5:16 like just saying hi and make sure this works as expected.
5:19 We can see
$\underline{5: 19}$ the model here is going to let us know not only that it's an AI assistant,
$\underline{5: 22}$ but also let us know some of the tools that it has available.
5:25 This is excellent.
5:26 So let's go ahead and start making use of these tools.
5:29 Let's search for recent papers on a topic of interest.
5:32 So why don't we go and search for papers on algebra.
5:36 And this should make use of the tool that we have to go ahead and search
5:40 for papers.
5:41 We can see that topics being passed in and the results are saved here.
5:45 This is great.
5:46 It's even following up with would you like me to extract more detailed information.
5:49 So I'll go ahead and say yes please extract information
5:54 on the first two you found and summarize them both for me.
5:59 So what we're going to do is make use of that tool result that we got before.
6:02 And we're going to pass that in
6:04 and it's going to tell us which IDs it's interested in.
6:06 So I'm going to make sure
6:07 that I pass in those particular IDs so I can get that correctly done.
6:11 The IDs are here.
6:13 So we're going to see here
$\underline{6: 13}$ it's going to extract the info with these particular IDs.
6:16 And we're going to go ahead and get the result as well as a summarization here.
6:19 So we got some information about in variant algebras and deformation of algebras.
6:24 Honestly, I cannot tell you what this is.
6:26 But hopefully I can read the paper
6:28 and get up to speed on what this information has.
6:30 Something to remember is that there is no persistent memory here,
6:33 so as you go ahead and search for queries and get IDs,
6:36 nothing is going to be stored permanently.
6:38 So just make sure as you keep querying, you're passing in those IDs
6:42 and think of each conversation as brand new each time.
6:45 If you ever want to get out of this chat, remember we can always type in quit.
6:48 So make sure you run that and we'll see that we're all done here.
6:51 So what we've seen in this lesson is a review of large language
6:54 models, tool use and making use of the archive SDK.
6:58 What we're going to see shortly is how we can refactor this code
7:01 to turn those tools into MCP tools
7:04 to allow for a server to pass that information to us.
7:07 We're then going to test that server and see what the results look like.
7:10 I'll see you in the next lesson.