---
layout: default
title: A kitchen that responds to your “I’m hungry”
nav_order: 6
description: ".."
has_children: false
parent:  Coursera - Microsoft Semantic Kernel
---
So we're coming close to the end. 
I know you're going to miss me, but you might not miss 
this kitchen metaphor, but I hope it works for you because everybody 
wants to be a chef. 
You're an AI chef. 
Are you feeling it? 
Okay, good. 
Keep coming with me. 
So what we're going to do in this section is to 
do something really hard. 
We're going to imagine a world where you just say I'm hungry 
and the LLM is able to complete that meal, use the 
plugins you've created and voila. 
You didn't have to do anything at all. 
You just kind of wished it. 
Let's jump into how that works. 
It's pretty amazing. 
Think of this small business owner. 
I like the two buckets. 
The bucket of time is leaking. 
The bucket of money is leaking. 
You know, do they have time to think about how 
to solve their business problems? 
Heck no. 
Now you have the tools to be able to give them interesting solutions, 
but at the same time, if they could just say, 
I wish I could do X and just have the 
AI help the business owner with as little effort as possible. 
For instance, let's say that the wish were something like, I 
wish I was $10 richer. 
I will need to blank. 
You know about the completion engine now, right? 
It'll complete it, but it's going to hallucinate because it's making stuff 
up. 
What do you do? 
You use retrieval augmented generation to 
kind of find similar things from your knowledge base. 
And then you might have different things that plugins could do, 
like write marketing copy or send an email. 
You can make plugins that do all kinds of things, 
native or semantic. 
And once you do that, what happens is each 
plugin will have a description of some form. 
And all you have to do is use that similarity engine to 
find the different tools in your tool bucket. 
Pause for a second. 
Anybody who has like a drawer full of 
kitchen tools knows that it's hard to find the whatchamacallit, but 
what if the AI can just say, I need something that opens 
cans and then voila, the tool appears. 
That's what happens when you have the similarity engine. 
You can magnetize all your plugins. 
If you had 2000 plugins, you as a human do not want to 
like use every plugin yourself. 
You want the AI to find the right plugin. 
So the third step would be finding the 
relevant plugins and then use those to be 
used in the completion response. 
That's kind of abstract. 
Remember, VAT of plugin, pull out the similar ones, 
use it for completion, push them to the kernel. 
And when you have this thing called a planner, 
the planner does that for you. 
And luckily in Semantec Kernel, we love plugins. 
And now you're learning that we love planners because if you 
got a lot of plugins, you'll need a planner. 
So let's do our quick inventory exercise, which 
you know I like, I'm a Wes Anderson fan. 
Okay, so we did design thinking, and then you just did 
a lot of good stuff. 
if you were able to use the similarity engine to your heart's 
content. 
Wow, you feel good. 
You look so good. 
Okay, that's you. 
Now we're gonna go into this notebook, which 
is the big one. 
First off, let's make a kernel, shall we? 
Let's make a kernel. 
This is the, you know how much I like the fire. 
The fire is what it feels like, because it basically is, you know, 
some GPUs, NPUs. 
Let's make a kernel one more time. 
You're going to miss me because we're not going to make kernels together. 
But you know, we're making kernels like, you know, I 
mean, it's okay. 
Let's just keep making kernels together. 
We can imagine that we're still making kernels together. 
You're doing great, by the way. 
So you know, it's not easy to do what you're doing. 
Okay, we just made the kernel. 
We ran it. 
You know how much, you know how much I like the print message, you know, 
I'm pretty old school print. 
And I, I did it boss, nothing like, you know, nothing like your co-pilot 
calling you boss, feel a little superior, nice job AI, nice 
job computational engine. 
Okay, next thing we're gonna do is remember the, let's take 
notes here for a second. 
We want to have a VAT of plugins and then find the 
right plugin to fit the goal. 
Right? 
So how do we do that? 
Well, there's different kinds of planners. 
Remember the planners, different kind of planners. 
And the reality is there is a super simple planner that people 
kind of make fun of. 
It's called the Action Planner. 
I admit that I had some kind of prejudice for the Action 
Planner and it fully appreciate it. 
But now I do because the Action Planner 
is just not that smart. 
In the Action Planner, you create it from the kernel. 
You give it a bunch of skills, plugins, sorry. 
If you notice Semantic Kernel has like this, 
like a complex where there's skills, but just call them plugins, work 
with me here. 
It's gonna shift over to that. 
But what I'm doing right now is I'm adding the tools for the kernel 
to do math, to read files, to tell 
the time and to play with text. 
Okay, so basically I didn't really do much besides 
add those tools into essentially a vat of 
plugins like I promised. 
And next up, let's do the thing that most people say, I 
was listening to a podcast where people were shading GPT-4, 
 
saying like, oh, well, you know, GPT-4 
is really stupid because it can't add. 
Well, I mean, like, I couldn't add when I was a kid, basically. 
So let's get over that, right? I mean, if 
you gave it the tool to add, you're going to give people a calculator. 
What happens? 
They start calculating, right? 
So we're going to get it to do math. 
And what we're going to do is we're going to use the planner to 
create a one, mind you, a one function, a single 
function that's gonna be pulled out of a vat of 
plugins to use. 
And what it's doing is it's taking this ask, and this is basically looking 
through all the available plugins it has available to 
it, skills, functions, et cetera, okay? 
And what I'm gonna do is I'm gonna ask it to tell me 
what that function is, right? 
All right, let's add some more print statements around this. 
Because programming is pretty abstract unless it tells you 
what you're doing. 
So let's run this. 
And so it's finding the most similar function available 
to get that done. 
What did it do? 
Wow, it knew that if I'm trying to get the sum of two numbers, it 
found in the math plugin, the addition function. 
How did it find it? 
Well, remember we made a description for each function. 
It's just comparing this question into the 
function description. 
Not a surprise, is it? 
Like, let's say like a, what is today? 
It's gonna look through the available plugins and it 
found in the time plugin, the today function. 
Do you see how that's working? 
It, you know, if you totally do something very complex, 
like what is the way to get to San Jose 
when the traffic is really bad? 
Now, this might require many plugins to work in concert, 
but as you can see, it's like, no, I really can't do that, boss, you know? 
 
So for simple things, how do I write the word text to a file? 
It's probably going to find in the file IO skill, it 
found the write function. 
Pretty cool, right? 
Again, it's very limited. 
It's no insult to you, computer. 
It's just not that smart. 
But when you can do that, a simple planner, 
you can imagine a planner that is much more powerful. 
And so the action planner is good for 
like a very basic searching through, find just one function. 
But what if I wanted to do a multi-step plan that's automatically 
generated, right? 
Let's do that. 
So what we're gonna do is we're going to pull in the 
sequential planner. 
The sequential planner is our gen two planner. There's 
a gen three planner that is, it's been ported from C-sharp. so it'll be coming in 
the repo shortly. 
Again, this is all open source, so you can get access to the latest 
and greatest as it comes out, fresh out of our kitchens 
to go into your kitchen. 
And all I'm gonna do is I'm going to bring in the literate friend 
plugin that I have. 
The literate friend plugin has a few functions. 
One, it can write poetry, it can translate, but I'm 
gonna hold onto that. 
Look at that, I'm still like, what a plugin. 
And then what I'm gonna do is I'm gonna make a sequential planner. 
Remember before I made an action planner, which wasn't too smart? 
Well, it's a sequential planner. 
It is. 
I would say like, not just a little, quite a lot smarter. 
I want it to do the following. 
Tomorrow is Valentine's day. 
I need to come up with a poem. 
Translate the poem to French. 
And so this is essentially gonna require two 
plugins, essentially one that can write the poem and 
the other that can translate. 
And I'm gonna do, I'm gonna call the planner. 
Oops, I forgot to close the string. 
That's red, right? 
Okay, there we go. All set? 
Good. 
All right, so we're gonna create plan, async. 
You know, we're built, we're architected in C-sharp where people 
ask, why is there a wait and like async everywhere? 
 
You know, this is enterprise software, 
people doing stuff. 
So I apologize, but in the end, you will thank us for all 
of our attendance to things that can happen asynchronously 
because we live in a network world, right? 
Okay, so that's gonna do that. 
And let's basically, let's print out the plan steps. 
I have that over here in my plated form. 
So let's see what happens. 
I'm going to bring in the literate friend plugin. 
It's got three functions. 
One is able to write a poem, one is able to translate. 
I think the other one is to summarize something, 
but there's three of them in there. 
I'm gonna ask it to make a plan to address this ask and let's see 
what happens. 
So if things work out the way we want, it's gonna realize that I 
need to write a poem and I need to translate it and 
so it pulled out two functions to use and you're like 
well great well can you use them absolutely so what I want 
to do is see what happens what happens when I have it 
actually tell me what it created and it 
says tomorrow's the Valentine's. 
I need to come up with a poem, that's my ask. 
It made a poem and then it translated it to French. 
Now let's do that in super slow motion, shall 
we? 
Let's print out the results step-by-step. 
And that's quite beautiful. 
Over here, okay. 
So I have a little trace results. 
You can look at the code later, but I'm gonna step through the 
plan and look at different things inside it 
and look at the input variables and output 
variables as they change. 
That is the weirdest part. 
So I'm gonna run that, and what you'll be able to see is that as 
the kernel takes the plan, the plan has already built a way to take the 
output from one and stuff it into the input of another, the 
plan has already built a way to take the output from one and 
stuff it into the input of another. 
Now watch this move here. 
The poem has been created and the poem's been created and it figured out 
to add a parameter French. 
Wow, isn't that amazing? 
So it basically plucked out the fact that I needed to make 
it in French and it took the poem output and there you have it. 
That is an automatically generated thing. 
Now, you may not think this is a big deal, 
but it's kind of a big deal because I did not have 
to tell the system to use those two plugins. 
I just gave it a box of plugins and it 
just went and pulled out the ones that need it. 
And number two, it created a plan, a multi-step plan to affect a 
more complex outcome. 
And wait for it. 
How does this work? 
I forget your magical piece of, oops, I forget. 
Let's see here. 
Markdown. 
I forget that there's two dimensions. 
There is completion. 
The completion is generating the plan. 
Similarity is pulling in context for the completion 
to be more right. 
And it's also pulling out the right plugins through the 
descriptions to be able to execute a plan. 
And what does this mean? 
It means that in the future, you won't 
go into kitchen and make dish one, dish two, dish three. 
You'll just say I'm hungry and it'll make you maybe five dishes if it 
has the right ingredients to do that. 
It'll make it the ones that you want. 
Because it's doing completion, it has similarity, understands 
what you have done in the past, and 
it can find the tools in the kitchen to go 
ahead and cook that and also the ingredients too. 
Okay, so that was a lot covered in a short amount of time. 
You must be exhausted by now. 
But I encourage you to play around with 
the different parameters here. 
Add more functions. 
Now that you know about the packaged format of functions, 
semantic functions, go ahead and throw 
in some native functions, which are by nature deterministic. 
Be sure to label all the variables, label all the descriptions, 
because as you now know, this similarity engine thing 
drives a lot of things. 
So it's the magic that lets the AI navigate your 
data to be able to augment the completion they go hand in 
hand left hand right hand together pretty amazing 
now we're gonna go to the conclusion of this course I'm gonna 
miss you but wait till you see what's up if 
you stick around for the next lesson 
