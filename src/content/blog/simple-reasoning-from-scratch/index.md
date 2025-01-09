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

From here, we can go about creating an environment for solving math problems. Environments are inspired by the [Gymnasium library](https://github.com/Farama-Foundation/Gymnasium), which abstracts MDPs using environment classes that support receiving observations from environments. This was similarly done by both [Openr](https://github.com/openreasoner/openr) and [TSLLM](https://github.com/waterhorse1/LLM_Tree_Search). We can first define a math problem dataclass which supports a question string, answer string, and an optional image  that's base64 encoded.

```python
@dataclass
class Problem:
    """A dataclass representing a problem"""
    question: str 
    answer: str 
    image: Optional[str] = None
```

 Since we're using the environment to support Chain-of-Thought reasoning, we can name it the `ChainOfThoughtEnv`. This class will be an abstract base class as we can implement the reward for a given dataset. This allows us to support arbitrary reward engineering. In the simplest case, reward can be 0 or 1 for a question being answered incorrectly or correctly, respectively. However, you may wish to engineer the reward to penalize longer chains of thought. Another uniquely engineered reward may be if you are training agentic reasoning, and wish to evaluate a path holistically rather than solely based on it's terminal state. 

```python 
class ChainOfThoughtEnv(ABC):
    """
    An environment for solving natural language problems using Chain-of-Thought (CoT) reasoning.
    
    This class implements a structured environment for solving problems (particularly math problems)
    using language models with Chain-of-Thought prompting. It maintains the state of the problem-solving
    process, manages interactions with the language model, and tracks the solution progress.

    Attributes:
        sep (str): Separator string used for joining text elements, and identifying completion of thoughts
        action_history (Optional[List[str]]): List of actions/steps taken in the current solution
        legal_actions (List[str]): List of valid next steps that can be taken
        image (Optional[str]): Image associated with the problem (if applicable) encoded as a base64 image

    Example:
        >>> llm = LLMGen("llama3.2")
        >>> env = CoTEnv(
        ...     problems=[problem1, problem2],
        ...     llm_gen=llm,
        ...     task_description="Solve these math problems step by step",
        ...     cot_examples="Example1...",
        ...     problem_format="Problem: {question}\nSolution:",
        ... )
        >>> state, reward, terminated, truncated, info = env.reset()
        >>> while not terminated or truncated:
        ...     action = llm(state)
        ...     state, reward, terminated, truncated, info = env.step(action)
    """
    sep: str = "\n"
    action_history: Optional[List[str]] = None
    legal_actions: List[str] = [] 
    image: Optional[str] = None 

    def __init__(
        self,
        problem: Problem,
        llm_gen: LLMGen,
        task_description: str,
        cot_examples: str,
        problem_format: str,
        stop_string: str,
        max_actions: int = 2,
        max_steps: int = 10,
        is_few_shot: bool = True,
        reset: bool = False,
    ):
        """
        Initialize the CoT environment.

        Args:
            problem: Problem to be solved
            llm_gen: Language model generator function/class for generating steps
            task_description: Description of the task to be solved
            cot_examples: Example problems and solutions for few-shot learning
            problem_format: Format string for presenting problems (must contain {question})
            stop_string: String specifying that the LLM is done it's chain of thought
            max_actions: Maximum number of actions to generate per step
            max_steps: Maximum length of solution steps
            is_few_shot: Whether to use few-shot learning with examples
            reset: Whether to reset the environment upon initialization
        """
        self.problem = problem    
        self.llm_gen_fn = llm_gen
        self.is_few_shot = is_few_shot
        self.max_actions = max_actions
        self.max_steps = max_steps

        self.task_description = task_description
        self.cot_examples = cot_examples
        self.problem_format = problem_format
        self.stop_string = stop_string

        prefixes = []
        if self.task_description is not None:
            prefixes.append(self.task_description)
        if self.is_few_shot:
            prefixes.append(self.cot_examples)
        if len(prefixes) > 0:
            self.task_prefix = self.sep.join(prefixes)
        else:
            self.task_prefix = None

        if reset:
            self.reset()

    def build_query_str(
        self,
    ) -> str:
        """
        Builds a formatted query string for the problem.

        Combines task description, examples (if few-shot), and the problem input
        into a formatted string ready for the language model.

        Returns:
            str: Formatted query string

        Example:
            >>> query = CoTEnv.build_query_str(
            ...     "Solve step by step",
            ...     "Example: 2+2=4",
            ...     "Problem: {question}",
            ...     "What is 3+3?",
            ...     True
            ... )
        """
        ret = ""
        if self.task_description:
            ret += self.task_description + "\n"
        if self.is_few_shot:
            ret += self.cot_examples + "\n"
        ret += self.problem_format.format(question=self.problem["question"])
        return ret

    def reset(self) -> Tuple[Tuple[str, Optional[str]], float, bool, bool, dict]:
        """
        Resets the environment to its initial state.

        Sets up a new problem, clears action history, and optionally updates legal actions.

        Returns:
           Tuple containing:
            - Current state (Tuple[str, Optional[str]])
            - Reward (float)
            - Whether episode is terminated (bool)
            - Whether episode is truncated (bool)
            - Additional info dictionary

        Raises:
            ResetException: If unable to establish legal actions after 3 attempts
        """
        self.image = self.problem.get("image", None)
        self.action_history = [
            self.build_query_str()
        ]
        state = self.get_state()
        return state, 0., False, False, {}
        
    def step(self, action: str) -> Tuple[Tuple[str, Optional[str]], float, bool, bool, dict]:
        """
        Takes a step in the environment by applying an action.

        Args:
            action: The action to take (next solution step)

        Returns:
            Tuple containing:
            - Current state (Tuple[str, Optional[str]])
            - Reward (float)
            - Whether episode is terminated (bool)
            - Whether episode is truncated (bool)
            - Additional info dictionary
        """
        self.action_history.append(action)
        state = self.get_state()
        reward = self.get_reward()
        terminated, truncated, info = self.get_done_and_info()
        return state, reward, terminated, truncated, info

    def get_state(self) -> Tuple[str, Optional[str]]:
        """
        Returns the current state of the environment.

        Returns:
            Tuple containing:
            - String representation of current state (action history)
            - Image associated with the problem (if applicable) encoded as a base64 image
        """
        return "\n".join(self.action_history) + "\n", self.image

    def get_done_and_info(self) -> Tuple[bool, bool, dict]:
        """
        Determines if the current episode is complete and provides additional information.

        Returns:
            Tuple containing:
            - Whether the episode is terminated (reached stop condition)
            - Whether the episode is truncated (reached max length)
            - Info dictionary with additional details (including winner)

        Note:
            winner codes:
            0: ongoing
            1: successful completion
            2: unsuccessful completion
        """
        info = {"winner": 0}
        terminated = self.stop_string in self.action_history[-1]
        max_steps = self.max_steps + (2 if self.task_prefix is not None else 1)
        
        truncated = len(self.action_history) >= max_steps
        assert len(self.action_history) <= max_steps, (
            f"action history length: {len(self.action_history)}, "
            f"max length: {max_steps}"
        )

        if terminated or truncated:
            info["winner"] = self.is_valid()
        
        return terminated, truncated, info

    @abstractmethod 
    def is_valid(self) -> Literal[1,2]:
        pass

    @abstractmethod
    def get_reward(self) -> float:
        pass 
```

An example environment that we can implement is for the [Grade School Math](https://github.com/openai/grade-school-math) dataset. 

```python 
class GSM8KEnv(ChainOfThoughtEnv):
    def is_valid(self) -> Literal[1,2]:
        """Extract the answer using regex, and then check it against the ground truth"""
        answer_regex = re.compile(r"The answer is (\-?[0-9\.\,]+)")
        match = answer_regex.search(self.action_history[-1])

        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "") or 'inf'
            is_correct = float(match_str) - float(self.problem["answer"]) < 1e-8
            return 1 if is_correct else 2
            
        else:
            match_str = ""
            return 2

    def get_reward(self) -> float:
        """Extract the answer using regex, and then check it against the ground truth"""
        answer_regex = re.compile(r"The answer is (\-?[0-9\.\,]+)")
        match = answer_regex.search(self.action_history[-1])

        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "") or 'inf'
            is_correct = float(match_str) - float(self.problem["answer"]) < 1e-8
            return 1. if is_correct else 0.
            
        else:
            match_str = ""
            return 0.
```

To work with the above environment in a typical RL scenario, we can use the following example code. Please note that we have to extract the numerical answer from the dataset as Grade School Math includes reasoning for answers already, followed by the actual answer after the string "#### "

```python 
llm_gen =  LLMGen("llama3.2:3b")
dataset = load_dataset("openai/gsm8k", 'main')

question = dataset['test']['question'][42]
answer = dataset['test']['answer'][42].split("#### ")[-1]

env = GSM8KEnv(
    problem = {"question": dataset['test']['question'][0], "answer": dataset['test']['answer'][0], "image": ""},
    llm_gen = llm_gen,
    task_description="Answer the following math questions",
    problem_format = "Question: {question}\nAnswer: Let's think step by step",
    cot_examples = (
        "Question: There are 15 trees in the grove. Grove workers will plant trees"
        "in the grove today. After they are done, there will be 21 trees. How many"
        "trees did the grove workers plant today?\nAnswer: Let's think step by step"
        "\nThere are 15 trees originally.\nThen there were 21 trees after some more"
        "were planted.\nSo there must have been 21 - 15 = 6.\nThe answer is 6\n\n"
        "Question: If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?\nAnswer: Let's think step by step\n"
        "There are originally 3 cars.\n2 more cars arrive.\n3 + 2 = 5.\nThe answer "
        "is 5\n\nQuestion: Leah had 32 chocolates and her sister had 42. If they ate"
        "35, how many pieces do they have left in total?\nAnswer: Let's think step "
        "by step\nOriginally, Leah had 32 chocolates.\nHer sister had 42.\nSo in "
        "total they had 32 + 42 = 74.\nAfter eating 35, they had 74 - 35 = 39.\nThe "
        "answer is 39\n\nQuestion: Jason had 20 lollipops. He gave Denny some "
        "lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to "
        "Denny?\nAnswer: Let's think step by step\nJason started with 20 lollipops."
        "\nThen he had 12 after giving some to Denny.\nSo he gave Denny 20 - 12 = 8."
        "\nThe answer is 8\n\nQuestion: Shawn has five toys. For Christmas, he got "
        "two toys each from his mom and dad. How many toys does he have now?\nAnswer:"
        "Let's think step by step\nShawn started with 5 toys.\nIf he got 2 toys each "
        "from his mom and dad, then that is 4 more toys.\n5 + 4 = 9.\nThe answer is 9"
        "\n\nQuestion: There were nine computers in the server room. Five more "
        "computers were installed each day, from monday to thursday. How many computers"
        " are now in the server room?\nAnswer: Let's think step by step\nThere were "
        "originally 9 computers.\nFor each of 4 days, 5 more computers were added.\n"
        "So 5 * 4 = 20 computers were added.\n9 + 20 is 29.\nThe answer is 29"
    ),
    stop_string = "The answer is"
)
```

To actually execute the environment, we can run the following code. If the LLM gets the correct answer, we should see a reward of one. Otherwise, the reward will be zero.

```python 
state, reward, terminated, truncated, info = env.reset()

while True: 
    response = llm_gen(state[0]).response
    state, reward, terminated, truncated, info = env.step(response)
    if terminated or truncated:
        print(env.get_state(), reward)
        break
```