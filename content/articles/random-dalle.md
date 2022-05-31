---
title: 🤖 Randomness in OpenAI's DALL-E 2
description: In this article, we explore feeding varying levels of randomness to OpenAI's DALL-E 2 text-to-image algorithm.
slug: random-dalle
---

I've gone ahead and provided the code to replicate these results at the bottom of the document.

# Random Words
The first method that we'll explore is feeding random words to DALL-E 2. In order to do so, we'll work with the Python Natural Language Toolkit `nltk`. Since we'll want to randomly sample words, we'll want to also import `random` from the base library. We can also download a list of all English words (236736 in total) provided by `nltk`, and set a random seed for reproducability. First, we'll sample some random words. This is probably the least likely to confuse DALL-E since words are relatively straightforward. 

![researchful](/article7/image001.png)

![researchful](/article7/image003.png)

![researchful](/article7/image005.png)


We can go ahead and sample variable length phrases ranging from three to eight words. This amount was chosen rather arbitrarily by myself, but more random words seems to be overkill. Using the code below, we can generate a random phrase. 

![researchful](/article7/image007.png)

![researchful](/article7/image009.png)

![researchful](/article7/image011.png)

# Random Characters 

![researchful](/article7/image013.png)

![researchful](/article7/image015.png)

![researchful](/article7/image017.png)

And then, some sample variable length phrases

![researchful](/article7/image019.png)

![researchful](/article7/image021.png)

![researchful](/article7/image023.png)

# Random Numbers
Some random characters 
![researchful](/article7/image025.png)

![researchful](/article7/image027.png)

![researchful](/article7/image029.png)

And now for some random longs

![researchful](/article7/image031.png)

![researchful](/article7/image033.png)

![researchful](/article7/image035.png)

![researchful](/article7/image037.png)


# Random Alphanumeric 

![researchful](/article7/image039.png)

![researchful](/article7/image041.png)

![researchful](/article7/image043.png)


And now som random alphanumeric phrases 

![researchful](/article7/image045.png)

![researchful](/article7/image047.png)

![researchful](/article7/image049.png)

