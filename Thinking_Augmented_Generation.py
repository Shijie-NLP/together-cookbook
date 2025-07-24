#!/usr/bin/env python
# coding: utf-8

# # Thinking Augmented Generation
# Author: [Zain Hasan](https://x.com/ZainHasan6)
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Agents/Thinking_Augmented_Generation.ipynb)

# ## Introduction
#
# In this notebook we will explore how you can improve the quality of smaller specialized models by using reasoning models.
#
# Specifically we will get `DeepSeek-R1` to reason about a prompt and then provide the `thinking` tokens to a smaller model like `Mistral Small 3` to generate a better response.
#

# In[1]:

import os

from together import Together


client = Together(api_key=os.getenv("TOGETHER_API_KEY"))


# Small model alone:

# In[27]:


question = "How many r's are in the word strawberry and burberry combined?"

answer = client.chat.completions.create(
    model="mistralai/Mistral-Small-24B-Instruct-2501",
    messages=[{"role": "user", "content": question}],
)


print(answer.choices[0].message.content)


# Let's get R1 to think about the question and then provide the thinking tokens to a smaller model like `Mistral Small 3` to generate a better response.
#

# In[ ]:


thought = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[{"role": "user", "content": question}],
    stop=["</think>"],  # Stop generation when </think> is encountered
)

print(thought.choices[0].message.content)


# Prompt template to pass in the thinking tokens to the smaller model:
#

# In[29]:


PROMPT_TEMPLATE = """
Question: {question}
Thought process: {thinking_tokens} </think>
Answer:
"""


# In[30]:


answer = client.chat.completions.create(
    model="mistralai/Mistral-Small-24B-Instruct-2501",
    messages=[
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(question=question, thinking_tokens=thought.choices[0].message.content),
        }
    ],
)

print(answer.choices[0].message.content)
