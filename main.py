from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from datasets import load_dataset
from haystack import Document


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


print(docs)
