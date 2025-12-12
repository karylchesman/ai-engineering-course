# System Retrieval Fundamentals

My notes during the AI ​​Engineering Specialization course.

# 3 Main Components

**Indexing** → Organized catalog of information that enables searching.

**Querying** → The process of searching through the index.

**Ranking** → Algorithms that determine how relevant an item in the index is to what is being searched for.

# Step-by-Step Process

1. **Data Collection** → Collecting data for indexing  
2. **Preprocessing** → Cleaning and normalizing the data  
3. **Indexing** → Creating the index using techniques that make search as efficient as possible, such as breaking terms, chunking, etc.  
4. **Querying** → Querying the index  
5. **Ranking** → Ranking the results based on relevance  
6. **Retrieval** → Presenting the data  

# Tokenization

The process of splitting text into smaller parts—words or sentences—that represent manageable units, enabling standardized information processing.

Basically 3 types: **word**, **sentence**, and **character** tokenization.

# Types of Systems (3 of them)

## Boolean Retrieval Model

The result is deterministic according to the filters. All or nothing. Either it matches the filters or it is not included in the results.  
Example of a Python library for full-text search with complex filters: **Whoosh**.

## Vector Space Model

Ranks results based on how “close” they are to what you’re looking for.

Example technique for this model: **TF-IDF (Term Frequency–Inverse Document Frequency)**.

## Probabilistic Retrieval Model

Ranks results based on the probability that an item is what you’re looking for, considering previously analyzed data.
