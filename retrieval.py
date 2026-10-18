"""Find the most relevant passages for a question."""

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from config import TOP_K


def retrieve(store: InMemoryVectorStore, question: str) -> list[Document]:
    # TOP_K limits the number of returned passages, even if the index contains
    # passages from many different PDFs. The store embeds the question using
    # the same embedding model that was used while building the index.
    retriever = store.as_retriever(search_kwargs={"k": TOP_K})
    
    # The returned Documents carry their text and original PDF/page metadata
    return retriever.invoke(question)
