import os
from getpass import getpass
from dotenv import load_dotenv
import pprint
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline


from datetime import datetime

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    
documents = [
    Document(
        content="Use pip to install a basic version of Haystack's latest release: pip install farm-haystack. All the core Haystack components live in the haystack repo. But there's also the haystack-extras repo which contains components that are not as widely used, and you need to install them separately.",
        meta={"version": 1.15, "date": datetime(2023, 3, 30)},
    ),
    Document(
        content="Use pip to install a basic version of Haystack's latest release: pip install farm-haystack[inference]. All the core Haystack components live in the haystack repo. But there's also the haystack-extras repo which contains components that are not as widely used, and you need to install them separately.",
        meta={"version": 1.22, "date": datetime(2023, 11, 7)},
    ),
    Document(
        content="Use pip to install only the Haystack 2.0 code: pip install haystack-ai. The haystack-ai package is built on the main branch which is an unstable beta version, but it's useful if you want to try the new features as soon as they are merged.",
        meta={"version": 2.0, "date": datetime(2023, 12, 4)},
    ),
]

# Initialize the document store     
document_store = InMemoryDocumentStore(bm25_algorithm="BM25Plus")

# Write documents to the store
document_store.write_documents(documents=documents)


# Initialize pipeline
pipeline = Pipeline()

pipeline.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="retriever")

query = "Haystack installation"
response = pipeline.run(
    data={
        "retriever": {
            "query": query, 
            "filters": {"field": "meta.version", "operator": ">", "value": 1.21}}})

pprint.pprint(response["retriever"])


response2 = pipeline.run(
    data={
        "retriever": {
            "query": query,
            "filters": {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.version", "operator": ">", "value": 1.21},
                    {"field": "meta.date", "operator": ">", "value": datetime(2023, 11, 7)},
                ],
            },
        }
    }
)

print("-----Response2-------")
pprint.pprint(response2["retriever"])