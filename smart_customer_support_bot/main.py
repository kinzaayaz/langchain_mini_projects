from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key
)

# Load documents
loader = WebBaseLoader("https://www.w3schools.com/python/python_intro.asp")
documents = loader.load()

# Split docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Embeddings + DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(texts, embeddings, collection_name="supabase_docs")
retriever = db.as_retriever(search_kwargs={"k": 3})

# Prompt template
prompt_template = """say greeting to user if user greets.
You are a support assistant. Use ONLY the provided CONTEXT to answer the QUESTION.
If the answer is NOT present in the CONTEXT, just reply "I don't know".

CONTEXT:
{context}
QUESTION:
{question}
Answer:"""

# Memory (for multi-turn chat)
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

# Conversational Retrieval Chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PromptTemplate(input_variables=["context", "question"], template=prompt_template)}
)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = qa.invoke({"question": user_input})
    print("AI:", response["answer"])
