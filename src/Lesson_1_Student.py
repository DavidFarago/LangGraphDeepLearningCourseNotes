import openai
import re
import httpx
import os
from dotenv import load_dotenv

_ = load_dotenv()
from openai import OpenAI

client = OpenAI()

# hello world test:
chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello world"}]
)
print(chat_completion.choices[0].message.content)
# 'Hello! How can I assist you today?'

class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
                        model="gpt-4o", 
                        temperature=0,
                        messages=self.messages)
        return completion.choices[0].message.content
    
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()

def calculate(what):
    return eval(what)

def average_dog_weight(name):
    if name in "Scottish Terrier": 
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")

known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}

# first example
abot = Agent(prompt)
result = abot("How much does a toy poodle weigh?")
print(result)
# Thought: I should look up the average weight of a Toy Poodle using the average_dog_weight action.
# Action: average_dog_weight: Toy Poodle
# PAUSE
result = average_dog_weight("Toy Poodle")
print(result)
# 'a toy poodles average weight is 7 lbs'
next_prompt = "Observation: {}".format(result)
print(abot(next_prompt))
# 'Answer: A Toy Poodle weighs an average of 7 lbs.'
print(abot.messages)
# [{'role': 'system',
#   'content': 'You run in a loop of Thought, Action, PAUSE, Observation.\nAt the end of the loop you output an Answer\nUse Thought to describe your thoughts about the question you have been asked.\nUse Action to run one of the actions available to you - then return PAUSE.\nObservation will be the result of running those actions.\n\nYour available actions are:\n\ncalculate:\ne.g. calculate: 4 * 7 / 3\nRuns a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary\n\naverage_dog_weight:\ne.g. average_dog_weight: Collie\nreturns average weight of a dog when given the breed\n\nExample session:\n\nQuestion: How much does a Bulldog weigh?\nThought: I should look the dogs weight using average_dog_weight\nAction: average_dog_weight: Bulldog\nPAUSE\n\nYou will be called again with this:\n\nObservation: A Bulldog weights 51 lbs\n\nYou then output:\n\nAnswer: A bulldog weights 51 lbs'},
#  {'role': 'user', 'content': 'How much does a toy poodle weigh?'},
#  {'role': 'assistant',
#   'content': 'Thought: I should look up the average weight of a Toy Poodle using the average_dog_weight action.\nAction: average_dog_weight: Toy Poodle\nPAUSE'},
#  {'role': 'user',
#   'content': 'Observation: a toy poodles average weight is 7 lbs'},
#  {'role': 'assistant',
#   'content': 'Answer: A Toy Poodle weighs an average of 7 lbs.'}]

# second example
abot = Agent(prompt)
question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
print(abot(question))
# 'Thought: I need to find the average weight of a Border Collie and a Scottish Terrier, then add them together to get the combined weight.\nAction: average_dog_weight: Border Collie\nPAUSE'
next_prompt = "Observation: {}".format(average_dog_weight("Border Collie"))
print(next_prompt)
# Observation: a Border Collies average weight is 37 lbs
print(abot(next_prompt))
# 'Action: average_dog_weight: Scottish Terrier\nPAUSE'
next_prompt = "Observation: {}".format(average_dog_weight("Scottish Terrier"))
print(next_prompt)
# Observation: Scottish Terriers average 20 lbs
print(abot(next_prompt))
# 'Thought: Now that I have the average weights of both dogs, I can calculate their combined weight by adding them together.\nAction: calculate: 37 + 20\nPAUSE'
next_prompt = "Observation: {}".format(eval("37 + 20"))
print(next_prompt)
# Observation: 57
print(abot(next_prompt))
# 'Answer: The combined weight of a Border Collie and a Scottish Terrier is 57 lbs.'

# add loop
action_re = re.compile('^Action: (\w+): (.*)$')   # python regular expression to selection action
def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a) 
            for a in result.split('\n') 
            if action_re.match(a)
        ]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return
question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
print(query(question))
# Thought: I need to find the average weight of a Border Collie and a Scottish Terrier, then add them together to get the combined weight.
# Action: average_dog_weight: Border Collie
# PAUSE
#  -- running average_dog_weight Border Collie
# Observation: a Border Collies average weight is 37 lbs
# Action: average_dog_weight: Scottish Terrier
# PAUSE
#  -- running average_dog_weight Scottish Terrier
# Observation: Scottish Terriers average 20 lbs
# Thought: Now that I have the average weights of both dogs, I can calculate their combined weight by adding them together.
# Action: calculate: 37 + 20
# PAUSE
#  -- running calculate 37 + 20
# Observation: 57
# Answer: The combined weight of a Border Collie and a Scottish Terrier is 57 lbs.