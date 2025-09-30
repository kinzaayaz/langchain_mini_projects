from langchain_groq import ChatGroq
from langchain.agents import tool
from langchain.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv
from langchain import hub
import os
import pandas as pd

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    streaming=False 
)

@tool
def file_reader(filepath: str = "transaction_data.csv") -> str:
    """
    Return dummy transaction data.
    """
    data = [
        [278166,6355745,"Sat Feb 02 12:50:00 IST 2019",465549,"FAMILY ALBUM WHITE PICTURE FRAME",6,11.73,"United Kingdom"],
        [337701,6283376,"Wed Dec 26 09:06:00 IST 2018",482370,"LONDON BUS COFFEE MUG",3,3.52,"United Kingdom"],
        [267099,6385599,"Fri Feb 15 09:45:00 IST 2019",490728,"SET 12 COLOUR PENCILS DOLLY GIRL",72,0.9,"France"],
        [380478,6044973,"Fri Jun 22 07:14:00 IST 2018",459186,"UNION JACK FLAG LUGGAGE TAG",3,1.73,"United Kingdom"],
        [285957,6307136,"Fri Jan 11 09:50:00 IST 2019",1787247,"CUT GLASS T-LIGHT HOLDER OCTAGON",12,3.52,"United Kingdom"],
        [345954,6162981,"Fri Sep 28 10:51:00 IST 2018",471576,"NATURAL SLATE CHALKBOARD LARGE",9,6.84,"United Kingdom"],
        [339822,6255403,"Mon Dec 10 09:23:00 IST 2018",1783845,"MULTI COLOUR SILVER T-LIGHT HOLDER",36,1.18,"United Kingdom"],
        [328440,6387425,"Sat Feb 16 10:35:00 IST 2019",494802,"SET OF 6 RIBBONS PERFECTLY PRETTY",36,3.99,"United Kingdom"],
        [316848,6262696,"Sat Dec 15 10:05:00 IST 2018",460215,"RED HARMONICA IN BOX",36,1.73,"United Kingdom"],
        [372897,6199061,"Mon Oct 29 09:04:00 IST 2018",459669,"WOODEN BOX OF DOMINOES",3,1.73,"United Kingdom"],
        [290111,6401234,"Tue Mar 05 14:30:00 IST 2019",500321,"CERAMIC TEAPOT WITH STRIPES",2,15.50,"France"],
        [301222,6423456,"Wed Mar 20 11:45:00 IST 2019",501234,"HEART SHAPED PHOTO FRAME",5,7.25,"United Kingdom"],
        [315678,6456789,"Fri Apr 12 16:00:00 IST 2019",502876,"VINTAGE BLUE LANTERN",1,22.10,"Germany"],
        [322333,6489012,"Mon Apr 29 13:10:00 IST 2019",503654,"SET OF 4 MASON JARS",12,4.75,"France"],
        [330999,6521345,"Sat May 18 17:20:00 IST 2019",504321,"CLASSIC WHITE DINNER PLATE",24,2.90,"Spain"],
        [340456,6554321,"Thu Jun 06 19:05:00 IST 2019",505987,"BAMBOO CUTTING BOARD",3,8.60,"Germany"],
        [350777,6587654,"Sun Jul 14 09:50:00 IST 2019",506543,"COZY KNIT BLANKET",2,35.00,"United Kingdom"],
        [362888,6612345,"Tue Aug 20 12:15:00 IST 2019",507111,"GLASS WATER BOTTLE",10,6.50,"France"],
        [375999,6645678,"Sat Sep 07 15:40:00 IST 2019",508432,"WOODEN SERVING TRAY",4,12.80,"Spain"],
    ]

    columns = [
        "UserId","TransactionId","TransactionTime","ItemCode",
        "ItemDescription","NumberOfItemsPurchased","CostPerItem","Country"
    ]

    df = pd.DataFrame(data, columns=columns)
    return df.to_string(index=False)


prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=[file_reader],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[file_reader],
    verbose=True,
    handle_parsing_errors=True
)

query = input("enter ur query here: ")
result = agent_executor.invoke({"input": query})
print(result['output'])
