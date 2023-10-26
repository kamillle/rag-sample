"""
Usage: poetry run python scripts/index_generator.py
"""

from llama_index import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("source_data").load_data()
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist()
