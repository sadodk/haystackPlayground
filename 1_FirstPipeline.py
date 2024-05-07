from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from datasets import load_dataset
from haystack import Pipeline
from haystack import Document
from haystack.components.builders import PromptBuilder
import os
from getpass import getpass
from haystack.components.generators import OpenAIGenerator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]


# Document store
document_store = InMemoryDocumentStore()


# Sentence transformer to embeds docs to vectors
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()

# Create embeddings of docs
docs_with_embeddings = doc_embedder.run(docs)

# Save embeddings to document store
document_store.write_documents(docs_with_embeddings["documents"])


# Initialize a text embedder (same type as doc embedder)
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

# Initialize a retriever (same type as document store)
retriever = InMemoryEmbeddingRetriever(document_store)


# Template for including the retrieved documents with the user query
template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

# Initialize prompt_builder
prompt_builder = PromptBuilder(template=template)

# Initialize generator
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key:")
generator = OpenAIGenerator(model="gpt-3.5-turbo")

# Initialize pipeline
basic_rag_pipeline = Pipeline()

# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", generator)


# Now, connect the components to each other
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")

# Start the loop to ask questions

while True:

    question = input("Ask a question: ")
    
    # Run the pipeline with the query embedding
    
    response = basic_rag_pipeline.run(
        {
            "text_embedder": {"text": question}, 
            "prompt_builder": {"question": question}
        }
    )

    print(response["llm"]["replies"][0])







