from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model= "llama-3.3-70b-versatile",
    groq_api_key = GROQ_API_KEY   
)

loader = CSVLoader(file_path="products.csv")
data = loader.load()

embeddings=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(data,embeddings)
retriver = vectorstore.as_retriever(search_kwargs={"k":3})

prompt = PromptTemplate(
    template="""
You are an intelligent E-commerce Product Recommender.

User request: {query}

Here are some candidate products:
{products}

Recommend the single best product that matches the request.
Just give a short explanation in 2–3 lines.
Add “reasoning chain” output → show why a product was recommended.
""",
    input_variables=["query", "products"]
)

parser = StrOutputParser()
chain = prompt | llm | parser

user_query = input("Enter your query here: ")
retrieved_docs = retriver.invoke(user_query)
retrieved_text = "\n".join(doc.page_content for doc in retrieved_docs)

response = chain.invoke({"query": user_query, "products": retrieved_text})

print("\n Recommendation with Reasoning \n")
print(response)