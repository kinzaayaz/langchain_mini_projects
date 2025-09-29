from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent,AgentExecutor
from langchain import hub
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm=ChatGroq(
    model= "llama-3.3-70b-versatile",
    groq_api_key = GROQ_API_KEY   
)

search = DuckDuckGoSearchRun()

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm = llm,
    tools = [search],
    prompt = prompt
)
agent_executor = AgentExecutor(
    agent = agent,
    tools = [search],
    verbose=False,
    handle_parsing_errors=True 
)

query = input("Enter your query: ")
result = agent_executor.invoke({"input":query})
print(result['output'])
