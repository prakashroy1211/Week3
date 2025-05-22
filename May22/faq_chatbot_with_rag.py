import chromadb
import asyncio
from typing import List, Dict, Any
import google.generativeai as genai

# Setup Gemini API (replace with your key)
genai.configure(api_key="AIzaSyDSU2A-L00m3ok1fVQREFdl6nrn5ajubuU")

# Setup ChromaDB client
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("faq_rag")

# Embedding function (dummy here, use actual embedding model)
def embed(text: str) -> List[float]:
    return [float(ord(c)) for c in text][:128] + [0.0] * (128 - len(text))  # Example 128-dim

# Add example documents
async def populate_faq():
    faq_data = [
    {"id": "1", "question": "How do I reset my password?", "answer": "Go to settings > account > reset password and follow the instructions."},
    {"id": "2", "question": "What is the refund policy?", "answer": "Refunds are processed within 30 days of purchase if eligibility criteria are met."},
    {"id": "3", "question": "How can I update my email address?", "answer": "Navigate to account settings and click on 'Update Email' to make changes."},
    {"id": "4", "question": "Can I use the app on multiple devices?", "answer": "Yes, your account can be accessed from multiple devices using your credentials."},
    {"id": "5", "question": "Why am I not receiving notification emails?", "answer": "Check your spam folder and ensure your notification settings are enabled."},
    {"id": "6", "question": "How do I delete my account?", "answer": "You can delete your account from settings > account > delete account. This action is irreversible."},
    {"id": "7", "question": "What payment methods are accepted?", "answer": "We accept credit cards, debit cards, PayPal, and UPI."},
    {"id": "8", "question": "How do I contact customer support?", "answer": "Click on the Help section in the app or email us at support@example.com."},
    {"id": "9", "question": "Is my data secure?", "answer": "Yes, we use end-to-end encryption and follow industry best practices for data security."},
    {"id": "10", "question": "How do I change my subscription plan?", "answer": "Go to Billing > Subscription and select the plan you want to switch to."},
    {"id": "11", "question": "Can I pause my subscription?", "answer": "Yes, you can pause your subscription anytime from your account settings."},
    {"id": "12", "question": "Why is the app crashing?", "answer": "Ensure you're using the latest version of the app. If the issue persists, contact support."},
    {"id": "13", "question": "Do you offer student discounts?", "answer": "Yes, students can apply for a 20% discount with a valid student ID."},
    {"id": "14", "question": "How do I enable dark mode?", "answer": "Go to Appearance settings and toggle the Dark Mode switch."},
    {"id": "15", "question": "Where can I find the user manual?", "answer": "The user manual is available in the Help section and can be downloaded as a PDF."},
    {"id": "16", "question": "How can I recover a deleted file?", "answer": "Check the trash or archive folder. If not found, contact support for recovery options."},
    {"id": "17", "question": "Can I schedule tasks in advance?", "answer": "Yes, use the Scheduler tool available in the dashboard to plan tasks."},
    {"id": "18", "question": "What happens if I miss a payment?", "answer": "Your access will be limited until payment is made. A reminder email will be sent."},
    {"id": "19", "question": "How do I enable two-factor authentication?", "answer": "Enable it from Settings > Security by linking your mobile number or authenticator app."},
    {"id": "20", "question": "How do I report a bug or error?", "answer": "Use the in-app feedback form or send an email with screenshots to bugs@example.com."}
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
    "Tell me about the refund rules.",
    "What are the accepted payment methods?",
    "How do I contact the support team?",
    "Can I access my account from multiple devices?",
    "How do I enable two-factor authentication?"
]

    for q in test_queries:
        print(f"User: {q}")
        result = await group_chat.chat(q)
        print(f"Bot: {result}\n")

# Run the async main
asyncio.run(main())