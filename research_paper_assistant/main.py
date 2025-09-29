from langchain_groq import ChatGroq
from langchain.agents import create_react_agent,AgentExecutor,load_tools
from langchain import hub
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model= "llama-3.3-70b-versatile",
    groq_api_key = GROQ_API_KEY
)
tool = load_tools(['arxiv'],)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm = llm,
    tools = tool,
    prompt = prompt
)

agent_executor = AgentExecutor(
    agent = agent,
    tools = tool,
    verbose = False,
    handle_parsing_errors = True
)

query = input("Enter paper name to search: ")
result = agent_executor.invoke({"input":query})
print(result['output'])
