from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm=ChatGroq(
    model= "llama-3.3-70b-versatile",
    groq_api_key = GROQ_API_KEY   
)

search = DuckDuckGoSearchRun()
query = input("Enter a news topic to search: ")
search_result = search.invoke(query)

prompt = PromptTemplate(
    template="""
    You are a strict news research assistant.

    - If the user query is about **current news or recent events**,shortly summarize the following search results in 3–5 key points.  
    Each point must include its source link in brackets.  

    - If the query is **not related to current news or events**, do NOT answer from your own knowledge.  
    Simply reply politely: "Sorry, I can only provide summaries for current news and events."

    user_query :{query}
    search result:{results}
    summary:

    """,
    input_variables = ["query","results"]
)

parser= StrOutputParser()
chain = RunnableSequence(prompt |llm |parser)
summary = chain.invoke({"query":query ,"results": search_result})
print(f"\n Summary for : {query}\n")
print(summary)



