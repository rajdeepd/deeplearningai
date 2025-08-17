---

---
0:01 MCP is based on a client-server architecture.
0:05 In this lesson, we'll go through the features that MCP can provide
0:08 and how the communication between the client and the server takes place.
0:12 All right. Let's dive in.
0:14 So we previously spoke about why the Model Context Protocol
0:17 is so useful for building Al applications and connecting to external data sources.
0:22 Now let's dive a little bit deeper into the architecture behind MCP.
0:26 Similar to other protocols, MCP follows the client-server architecture,
0:31 where we have MCP clients that maintain a 1 to 1 connection with MCP servers.
0:36 The way these two communicate with each other is through messages
0:40 defined by the MCP itself.
0:43 These clients live inside of a host.
0:46 This could be something like Claude desktop or Claude AI.
0:50 The host is responsible for storing and maintaining
0:53 all of the clients and connections to MCP servers.
0:56 We'll see this in a little more depth.
0:58 Hosts are LLM applications that want to access data through MCP.
1:02 The servers are lightweight programs that expose the specific capabilities
1:07 through the protocol.
1:08 And very soon we're going to start building our own servers.
1:10 We'll then build our own clients as well as hosts that contain multiple clients.
1:15 The code for that is going to be a little bit more lower level.
1:17 But the goal here is really to understand the architecture.
1:21 And when you use tools like Claude Desktop or Cursor
1:24 or Windsurf, you have an idea of what's happening under the hood.
1:27 So how does it work?
1:28 Before we discuss the responsibilities of the client and the server,
1:32 let's dive into some of the primitives or fundamental pieces of the protocol.
1:36 Starting with tools.
1:37 If you're familiar with tool use, tools in MCP are going to look very similar.
1:42 Tools are functions that can be invoked by the client.
1:45 These tools allow for retrieving, searching,
1:47 sending messages, and updating database records.
1:50 Tools are usually meant for data that might require
1:53 something like a Post request or some kind of modification.
1:56 Resources are a bit more similar to a Get request.
1:59 Resources are read-only data or context that's exposed by the server.
2:03 Your application can choose whether to consume
2:06 or use these resources, but it doesn't necessarily have to bring it into context.
2:10 Examples of resources can include database records, API responses,
2:15 files, PDFs, and so on that you may have.
2:18 The third primitive we're going to explore, is a prompt template.
2:20 And prompt templates aim to achieve a very reasonable task,
2:25 which is to remove the burden of prompt engineering from the user.

2:28 You might have an MCP server whose job is to query things in Google
2:32 Drive and summarize and so on, but the user itself would need to write
2:37 the prompt necessary
$\underline{2: 38}$ to achieve all of those tasks in the most efficient way possible.
2:41 Instead of mandating that the user write the entire prompt
$\underline{2: 45}$ and figure out the best practices for prompt engineering,
2:47 prompt templates are predefined templates that live on the server
$\underline{2: 52}$ that the client can access and feed to the user if they so choose.
$\underline{2: 56}$ We're going to see in a few lessons how to build tools, resources,
2:59 and prompt templates both on the server and the client.
3:02 The client's job is to find resources and find tools.
3:06 The server's job is to expose that information to the client.
3:10 Now that we have an idea of some of these primitives: tools,
3:13 resources, prompts, let's go explore what this actually looks like.
3:16 I'm going to be using a host Claude Desktop, and I'm going to connect
3:20 to an MCP server for SQL Lite that exposes tools, resources, and prompts.
3:25 So let's take a look at that right here in Claude Desktop
3:27 I've connected to an MCP
3:29 server and I have tools at my disposal to work with SQLite.
3:33 We'll talk a bit about the configuration settings in a later lesson.
3:36 I wanted to show you what this looks like in action.
3:39 Once I connect to this MCP server, I can start talking to my data
$\underline{3: 43}$ in natural language.
3:44 So I'll ask right after that what tables do I have
3:48 and how many records are in each table.
3:52 This right here is Claude connecting to the Outside World.
3:56 We can see here we're going to be using a tool
3:58 from the SQL light server called List Tables.
4:02 You can see in the request there's no dynamic data being sent.
4:05 And I'll go ahead and allow this.
4:07 The ability to require a human in the loop
4:10 is based on the interface that the host develops.
$\underline{4: 13}$ So the server itself is simply sending back the tools.
4:16 The client is then
4:17 taking advantage of those tools and executing the data necessary.
4:21 We can see here for the number of records that we have
$\underline{4: 23} 30$ products, 30 users and zero orders.
4:26 So we can see the records that we have in this table.
4:28 What we can start to do now is something a little more interesting.
4:31 By taking advantage of tools like artifacts
4:33 and making this slightly more visually appealing.
$\underline{4: 36}$ So generate an interesting visualization
4:41 based on
4:43 the data in the products table.
4:48 You can imagine even with my spelling
4:49 mistake, we'll be able to query that information that we need.

4:49 mistake, we'll be able to query that information that we need.
$\underline{4: 52}$ So we'll go find the table.
4:53 We'll run the necessary query and fetch the data necessary.
4:57 We'll see here we're going to analyze this.
4:59 And it's going to tell us
5:00 many things are price but there are a few higher-priced items.
5:03 We're going to use the analysis tool to analyze this data.
5:06 What we're bringing in here is context to an AI application.
5:10 This could be Claude desktop.
5:11 This could be with any other model.
5:13 This could be in any other environment.
5:15 But through MCP we can build really interesting
5:17 applications right off the bat.
5:19 I'm making use of the artifacts feature in Claude
5:22 so that we can see a nice little visualization.
5:24 But the goal here is really to let your imagination carry you
5:27 where you can go with this.
5:28 Bringing in external data.
5:30 External systems allows you to easily create much
5:33 more interesting, compelling, and powerful applications.
5:36 We'll see here the code generated.
5:37 It's going to be a nice little visualization.
5:39 I have a distribution.
5:40 I have price versus quantity and so on.
5:42 And right after that,
5:43 I can take this data that I want and turn it into something meaningful.
5:46 We're doing this through tool use.
5:48 So the first primitive that we've explored
5:50 are the tools given to us by the MCP server.
5:53 Next, let's explore some other primitives.
5:55 I'm going to see here in SQLite that there is an MCP demo prompt.
6:00 This is a prompt template that is being sent from the server,
6:02 where all I have to do as the user is pass in some dynamic data.
6:06 So here we're demonstrating what we can do with this particular prompt.
6:10 This is a topic to see the database with initial data.
6:13 So let's go ahead and seed the database with some data around planets.
6:17 When I add this to the prompt we can see right off the bat
6:20 there is a text file that's generated with a prompt
6:23 that the server has given me. We can see right here
6:26 this is not something that I, as the user, have to write.
6:29 I just choose the dynamic data and then I go ahead and run that particular prompt.
6:33 What we're going to see here is this prompt in action.
6:35 And here this is going to generate a business problem and analyze
6:38 some data and set up information and so on.
6:41 But you can imagine giving your users much more battle
6:44 tested evaluated prompts so you don't have to do it yourself.
6:48 You'll see here we're going to set up some tables.

6:48 You'll see here we're going to set up some tables.
6:50 We're going to query.
6:51 We're going to populate
6:52 all kinds of actions we can take based on the prompt
6:55 and the tools that we have at our disposal.
6:57 So here you're seeing an example of tools and prompts being integrated together
7:01 to make AI applications far more powerful than they are out of the box.
7:05 In a few lessons, we're going to start making our own prompts,
7:08 our own resources, and our own tools to see how this happens under the hood.
7:12 As we go through, we can actually see that there is a data insight here,
7:16 a business insight memo that gets updated as we are constantly adding more data.
7:20 This is an example of a resource.
7:22 Resources are dynamic.
7:23 They can be updated as data changes in your application.
7:27 And instead of requiring tools to fetch this information,
7:30 we have data here that can constantly be updated.
7:33 I could ask to update the memo.
7:35 I could ask to update information inside based on new data that I've achieved.
7:39 So in this little example we've seen a host Claude Desktop.
7:43 We've seen a variety of tools from the SQLite MCP server,
7:47 and we've seen prompts and resources that allow us to perform
7:49 really powerful actions.
7:51 Now that we've seen what it looks like to use tools with MCP
$\underline{7: 55}$ servers, let's go ahead and talk about how you actually create these.
7:58 MCP provides software development kits for building servers and clients
8:02 in quite a few languages.
8:03 In this course, you'll be using the Python MCP SDK,
8:06 which makes it very easy to declare tools, resources, and prompts.
8:11 You can see here to declare a tool.
8:13 We decorate a function.
8:14 We pass in the necessary arguments
8:16 and return values so that the tool schema can be generated.
8:20 And then we return
8:21 what happens when that tool needs to be executed.
8:23 For resources, we allow the server to expose data to the client.
8:28 And that's done by specifying a URI
8:30 or a location where the client goes to find that data.
8:34 You can call this whatever you want, but you can imagine to return a list
8:37 of documents, this is a pretty good one.
8:39 If you're sending back a certain data format,
8:41 you can specify that with the main type.
8:43 You decorate a function which returns the data that you want
8:47 when that resource is accessed.

8:48 And you can do that for direct resources.
8:51 Or if you happen to have some kind of dynamic information or ID,
8:55 you can go ahead and use a templated resource, just like an F string in Python.
8:59 To give you an example of what an interface
9:01 might look like with a resource.
9:03 You can have a command line application where you use an @ sign to then fetch
9:07 all the documents that you need,
9:08 or for a templated resource, you can reference that directly and inject
9:12 that into a prompt or request that's coming in. With resources,
9:16 we don't need tools to fetch the data that we need.
9:18 The server simply sends the data back to the client,
9:21 and the application chooses to use that data or not.
9:24 Lastly, let's talk about prompts.
9:26 Just like you saw
9:27 with decorating tools and resources, we do the same thing with a prompt.
9:31 We give it a name and description,
9:33 and then a list of messages or text to return back. With prompts,
9:38 you define a set of user assistant messages
9:41 or just the text of a prompt that you need.
9:43 You can imagine a situation where a user might want to convert
9:46 some data to markdown, and while this is a fine prompt,
9:50 it might be a lot nicer if you gave them a thoroughly evaluated prompt instead.
9:54 So with prompts and prompt templates,
9:56 the idea is for these to be user controlled,
9:59 where a user chooses to not have to do all the prompt engineering themselves,
10:02 and use the quality ones provided by the server.
10:05 Now that we have an idea on some of the primitives, let's talk
10:08 a little bit about the communication between clients and servers.
10:11 When the client opens up a connection to the server,
10:14 there's an initialization process where a request is sent, a response is
10:18 sent back, and a notification is sent to confirm initialization.
10:22 Once that initialization appears, there's an exchange of messages that happen.
10:26 It's important to look at these steps because in the code
10:29 you're actually going to see methods like initialize.
10:32 So make sure you understand these ideas under the hood.
10:35 So when we start writing code
10:36 you can understand what's happening. In message exchanges,
10:39 clients can send requests to servers.
10:41 Servers can send requests
10:43 to clients, notifications can also be sent back and forth.
10:46 We'll talk a bit later on
10:48 about some of the other protocols, where servers can sample
10:51 or request information from clients, and notifications can be sent both ways.
10:55 Finally, at the end of communication, there's a termination of that connection.

11:00 As we talk a little bit
11:00 more about the connection and the way in which messages are sent back and forth.
11:05 It's important to understand another part of the Model Context Protocol,
11:09 and that is the idea of a transport.
11:10 And a transport handles the mechanics of how messages are sent back and forth
11:15 between the client and the server,
$11: 16$ depending on how you're running your application.
11:19 You will choose one of these different transports.
11:22 You can also make your own if you would like. For servers running locally,
11:26 we're going to be using standard IO or standard input output.
11:29 When we start deploying servers remotely later on in the course, we have the choice
11:33 between using HTTP and server-side events
$11: 36$ or using the Streamable HTTP transport.
11:40 As of this time of recording, Streamable HTTP
$11: 43$ is not supported yet across all software development kits.
11:47 So we're going to be talking in depth about it.
11:48 But in our example we'll be using HTTP with server center events.
11:52 To give you a little bit of a distinction,
11:54 when you're using HTTP, which servers and events
11:57 you need to open up a stateful connection that maintains a back-and-forth
12:01 that's open.
12:02 For certain kinds of applications and deployments, or stateless deployments,
12:06 this does not suffice.
12:08 So in a newer version of the specification, the Streamable HTTP
12:12 transport allows for stateful connections as well as stateless.
12:16 To talk about our first transport standard, IO.
12:18 The process involves the client launching the server as a subprocess,
$12: 22$ and the server
$12: 23$ reading and writing alongside the client with standard in and standard out.
12:27 All of this is going to be abstracted away from us,
$12: 30$ but it's important to understand when using standard IO
$12: 33$ that this is most commonly what's done when running servers locally.
12:37 When we talk about the transports or remote servers, you'll see that
12:41 there are different transports based on different versions of the protocol.
12:44 The original transport for remote servers involved
12:47 using HTTP and server events for a stateful connection.
12:52 In a stateful connection, the client and the server are communicating
$12: 55$ with each other with no closed connection between requests.
12:59 Data can be shared.
13:00 Data can be sent and remembered between different requests.
13:04 With the ability to introduce state. In server sent events,
13:07 the server is also able to send back events and messages back to the client.
$13: 12$ While this can work for a variety of applications, many applications
$13: 16$ when deployed are not stateful nor need to be stateful.