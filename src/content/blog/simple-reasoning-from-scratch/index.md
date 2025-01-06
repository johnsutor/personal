---
title: "🤔 Simple LLM Reasoning from Scratch"
description: "Learn how to quickly and easily teach LLMs to reason, from scratch."
date: "2025-01-06"
---
## Introduction
Recently, there's been quite a bit of hype around teaching LLMs how to reason and learn. There's been an emergence of quite a few libraries or methods purpose-built for teaching models to reason ([Openr](https://github.com/openreasoner/openr), [Search and Learn](https://github.com/huggingface/search-and-learn), and [TRL's PRM Trainer](https://github.com/huggingface/trl), just to name a few). However, I've found that many of these are unapproachable for quickly hacking, given their dependencies for running LLMs in CUDA environments on-device. Furthermore, these libraries aren't necessarily extensively documented, making them somewhat unapproachable for hacking. Thus, I provide this tutorial for teaching LLMs to reason from scratch on a consumer-grade computer. 

## Getting Started 
For starters, we need to install some necessary libraries and tools. We'll be working with [Ollama](https://ollama.com/) for LLM inference, which is very straightforward to download on any desktop OS. Once you have Ollama installed, you can run the following to pull a lightweight Llama model. This will load the [3b model](https://ollama.com/library/llama3.2) by default. For an extensive list of models that you can use, you can [browse Ollama models](https://ollama.com/library).

```shell
ollama pull llama3.2
```

We'll also go ahead and set up a [Conda](https://anaconda.org/) environment with all required packages. If you don't use Conda, you can bypass the first two steps and install the libraries using a virtual environment in [UV](https://github.com/astral-sh/uv) or [Venv](https://docs.python.org/3/library/venv.html) as well. From there, we'll install the [Ollama](https://github.com/ollama/ollama-python), [Datasets](https://github.com/huggingface/datasets), and [NumPy](https://github.com/numpy/numpy) packages to interract with the Ollama client, load mathematical reasoning datasets, and perform vector operations, respectively. We'll also install jupyter to work with these tools in a Jupyter notebook.

```shell 
conda create -n local-reasoning python=3.10
conda activate local-reasoning 
pip install ollama datasets numpy 
```

## Creating an environment 
A useful abstraction for working with step-by-step problem solving is treating it like a [Markov Decision Process (MDP)](https://en.wikipedia.org/wiki/Markov_decision_process). Thus, we can create an environment that represents the state of learning how to solve a problem, and have our LLM act as an agent. For starters, we can create a utility function for calling our model, like below. Note that we forego setting more advanced sampling parameters such as min_p sampling for simplicity, but you can augment this code to support such sampling methods. We opt for a large temperature and top_p for a high diversity of sampled outputs. Importantly, we also specify that we wish to use raw prompting. When you call an LLM, a template is applied to the inputs. You can find this template on a given model's page on Ollama. For example, the Llama 3.2 template can be found [here](https://ollama.com/library/llama3.2/blobs/966de95ca8a6). However, if you use a template in this code, you'll see that the model will start from scratch with reasoning at every time step when we implement the environment class. Thus, we call `raw=True` to ensure that the template is not applied. Finally, you'll see that we're applying a stop token for calling the model. This is to specify that we wish to generate trees at a sentence-level, as is done in [this paper](https://arxiv.org/pdf/2309.17179). There are follow-up papers that propose using tokens to specify the ends of thoughts, though for simplicity and to use an out-of-the-box model, we will opt for sentence-level reasoning. 

```python 
import ollama 

class LLMGen:
    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 128
    ):
        """
        A generic wrapper function for calling Ollama with some default arguments. Pulls the specified model, and when calling this class, will generate a response given the sampling parameters without the use of a template.

        Args:
            model (str): The model we wish to use. Make sure to pull this model using `ollama pull {YOUR MODEL NAME HERE}` before calling this model. 
            temperature (float): The temperature to use when sampling. A lower temperature means less variability in outputs, whereas a higher temperature means higher variability in outputs. 
            top_p (float): The nucleus sampling method for determining what tokens to consider for sampling. 
            n (int): The number of tokens to predict in a given step. 
        """
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.n = n

        ollama.pull(model)

    def __call__(self, content: str, image: str = None):
        return ollama.generate(
            self.model, 
            content,
            images = [image] if image else None,
            options={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.n,
                "stop": ["\n"]
            },
            raw=True,
        )
```

You can quickly test this model using the code below 

```python 
LLMGen("llama3.2")("What is 2 + 2")
```

From here, we can go about creating an environment for solving math problems. Environments are inspired by the [Gymnasium library](https://github.com/Farama-Foundation/Gymnasium), which abstracts MDPs using environment classes that support receiving observations from environments. This was similarly done by both [Openr](https://github.com/openreasoner/openr) and [TSLLM](https://github.com/waterhorse1/LLM_Tree_Search). Since we're using the environment to support Chain-of-Thought reasoning, we can name it the `CoTEnv`