#!/usr/bin/env python
# coding: utf-8

# # Retrieval-Augmented Generation (RAG) with Llama3 8B
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/togethercomputer/together-cookbook/blob/main/Text_RAG.ipynb)

# ## Introduction
#
# For AI models to be effective in specialized tasks, they often require domain-specific knowledge. For instance, a financial advisory chatbot needs to understand market trends and products offered by a specific bank, while an AI legal assistant must be equipped with knowledge of statutes, regulations, and past case law.
#
# A common solution is Retrieval-Augmented Generation (RAG), which retrieves relevant data from a knowledge base and combines it with the user’s prompt, thereby improving and customizing the model's output to the provided data.
#
# <img src="images/simple_RAG.png" width="500">

# ## RAG Explanation
#
# RAG operates by preprocessing a large knowledge base and dynamically retrieving relevant information at runtime.
#
# Here's a breakdown of the process:
#
# 1. Indexing the Knowledge Base:
# The corpus (collection of documents) is divided into smaller, manageable chunks of text. Each chunk is converted into a vector embedding using an embedding model. These embeddings are stored in a vector database optimized for similarity searches.
#
# 2. Query Processing and Retrieval:
# When a user submits a prompt that would initially go directly to a LLM we process that and extract a query, the system searches the vector database for chunks semantically similar to the query. The most relevant chunks are retrieved and injected into the prompt sent to the generative AI model.
#
# 3. Response Generation:
# The AI model then uses the retrieved information along with its pre-trained knowledge to generate a response. Not only does this reduce the likelihood of hallucination since relevant context is provided directly in the prompt but it also allows us to cite to source material as well.
#
# <img src="images/text_RAG.png" width="750">

# ### Install libraries

# In[2]:


import os

import together
from together import Together


# Paste in your Together AI API Key or load it
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")


# In[5]:


import json


with open("./datasets/movies.json", "r") as file:
    movies_data = json.load(file)

print(movies_data[:3])


# ## Implement Retreival Pipeline - "R" part of RAG
#
# Below we implement a simple retreival pipeline:
# 1. Embed movie documents + query
# 2. Obtain top k movies ranked based on cosine similarities between the query and movie vectors.

# In[8]:


# This function will be used to access the Together API to generate embeddings for the movie plots

from typing import List

import numpy as np


def generate_embeddings(input_texts: List[str], model_api_string: str) -> List[List[float]]:
    """Generate embeddings from Together python library.

    Args:
        input_texts: a list of string input texts.
        model_api_string: str. An API string for a specific embedding model of your choice.

    Returns:
        embeddings_list: a list of embeddings. Each element corresponds to the each input text.
    """
    together_client = together.Together(api_key=TOGETHER_API_KEY)
    outputs = together_client.embeddings.create(
        input=input_texts,
        model=model_api_string,
    )
    return np.array([x.embedding for x in outputs.data])


# In[6]:


# Concatenate the title, overview, and tagline of each movie
# this makes the text that will be embedded for each movie more informative
# as a result the embeddings will be richer and capture this information.

to_embed = []
for movie in movies_data[:1000]:
    text = ""
    for field in ["title", "overview", "tagline"]:
        value = movie.get(field, "")
        text += str(value) + " "
    to_embed.append(text.strip())

print(to_embed[:10])


# In[9]:


# Use bge-base-en-v1.5 model to generate embeddings
embeddings = generate_embeddings(to_embed, "BAAI/bge-base-en-v1.5")


# In[10]:


# Generate the vector embeddings for the query
query = "super hero action movie with a timeline twist"

query_embedding = generate_embeddings([query], "BAAI/bge-base-en-v1.5")[0]


# In[11]:


from sklearn.metrics.pairwise import cosine_similarity


# Calculate cosine similarity between the query embedding and each movie embedding
similarity_scores = cosine_similarity([query_embedding], embeddings)


# In[12]:


# We get a similarity score for each of our 1000 movies - the higher the score, the more similar the movie is to the query
print(similarity_scores.shape)


# In[14]:


# Get the indices of the highest to lowest values
indices = np.argsort(-similarity_scores)


# In[15]:


top_10_sorted_titles = [movies_data[index]["title"] for index in indices[0]][:10]

print(top_10_sorted_titles)


# ### Retreiver Function
#
# Once we understand the steps in the retriever pipeline above we can simplify it into the function below.

# In[19]:


def retreive(query: str, top_k: int = 5, index: np.ndarray = None) -> List[int]:
    """
    Retrieve the top-k most similar items from an index based on a query.
    Args:
        query (str): The query string to search for.
        top_k (int, optional): The number of top similar items to retrieve. Defaults to 5.
        index (np.ndarray, optional): The index array containing embeddings to search against. Defaults to None.
    Returns:
        List[int]: A list of indices corresponding to the top-k most similar items in the index.
    """

    query_embedding = generate_embeddings([query], "BAAI/bge-base-en-v1.5")[0]
    similarity_scores = cosine_similarity([query_embedding], index)

    return np.argsort(-similarity_scores)[0][:top_k]


# In[20]:


print(retreive("super hero action movie with a timeline twist", top_k=5, index=embeddings))


# ## Generation Step - "G" part of RAG
#
# Below we will inject/augment the information the retreival pipeline extracts into the prompt to the Llama3 8b Model.
#
# This will help guide the generation by grounding it from facts in our knowledge base!

# In[22]:


# Extract out the titles and overviews of the top 10 most similar movies
titles = [movies_data[index]["title"] for index in indices[0]][:10]
overviews = [movies_data[index]["overview"] for index in indices[0]][:10]


# In[24]:


client = Together(api_key=TOGETHER_API_KEY)

# Generate a story based on the top 10 most similar movies

response = client.chat.completions.create(
    model="meta-llama/Llama-3-8b-chat-hf",
    messages=[
        {
            "role": "system",
            "content": "You are a pulitzer award winning craftful story teller. Given only the overview of different plots you can weave together an interesting storyline.",
        },
        {
            "role": "user",
            "content": f"Tell me a story about {titles}. Here is some information about them {overviews}",
        },
    ],
)

print(response.choices[0].message.content)


# Here we can see a simple RAG pipeline where we use semantic search to perform retreival and pass relevant information into the prompt of a LLM to condition its generation.
#
# To learn more about the Together AI API please refer to the [docs here](https://docs.together.ai/docs/introduction)!
