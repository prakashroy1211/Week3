import chromadb
import asyncio
from typing import List, Dict, Any
import google.generativeai as genai

# Setup Gemini API (replace with your key)
genai.configure(api_key="#")

# Setup ChromaDB client
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("faq_rag")

# Embedding function (dummy here, use actual embedding model)
def embed(text: str) -> List[float]:
    return [float(ord(c)) for c in text][:128] + [0.0] * (128 - len(text))  # Example 128-dim

# Add example documents
async def populate_faq():
    faq_data = [
        {"id": "1", "question": "How do I reset my password?", "answer": "Go to settings > reset password."},
        {"id": "2", "question": "What is the refund policy?", "answer": "Refunds are allowed within 30 days."}
    ]
    for item in faq_data:
        collection.add(documents=[item["question"] + " " + item["answer"]],
                       ids=[item["id"]],
                       embeddings=[embed(item["question"] + item["answer"])])

# RAG Retriever
class RAGRetriever:
    def __init__(self, collection):
        self.collection = collection

    async def retrieve(self, query: str, top_k: int = 3) -> str:
        embedded_query = embed(query)
        results = self.collection.query(query_embeddings=[embedded_query], n_results=top_k)
        docs = [doc for sublist in results["documents"] for doc in sublist]
        return "\n".join(docs)

# Gemini-based response agent
class GeminiResponder:
    async def respond(self, query: str, context: str) -> str:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = await asyncio.to_thread(model.generate_content, f"Context: {context}\n\nQuestion: {query}")
        return response.text

# Query Handler using RoundRobinGroupChat
class RoundRobinGroupChat:
    def __init__(self, agents):
        self.agents = agents
        self.index = 0

    async def chat(self, query: str) -> str:
        # Select agent in round robin
        agent = self.agents[self.index % len(self.agents)]
        self.index += 1
        return await agent.handle(query)

# Combined Query Handler
class QueryHandler:
    def __init__(self, retriever: RAGRetriever, responder: GeminiResponder):
        self.retriever = retriever
        self.responder = responder

    async def handle(self, query: str) -> str:
        context = await self.retriever.retrieve(query)
        response = await self.responder.respond(query, context)
        return response

# Setup everything
async def main():
    await populate_faq()
    retriever = RAGRetriever(collection)
    responder = GeminiResponder()
    query_handler = QueryHandler(retriever, responder)

    group_chat = RoundRobinGroupChat([query_handler])
    
    # Simulate some queries
    test_queries = [
        "How can I change my password?",
        "Tell me about the refund rules."
    ]

    for q in test_queries:
        print(f"User: {q}")
        result = await group_chat.chat(q)
        print(f"Bot: {result}\n")

# Run the async main
asyncio.run(main())