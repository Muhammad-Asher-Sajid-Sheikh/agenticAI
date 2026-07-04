import os
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


load_dotenv()
'''
if not os.environ.get("GEMINI-API-KEY"):
  print("Key not found")
  exit(1)
'''

GEMINI_API_KEY = "Asd12edw" # Add your Gemini API key here


embeddingmodel = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-V2")
qdrant_client = QdrantClient(":memory:")
collection_name = "degree_data"

data = [
    "Studying Software Engineering at NED University of Engineering and Technology in Karachi is known for offering a robust, engineering-focused curriculum",
    "Degree & Accreditation: The department offers a Bachelor of Engineering (B.E.) in Software Engineering, which holds Level-II accreditation from the Pakistan Engineering Council (PEC)",
    "Core Curriculum: The syllabus blends theoretical computer science with practical engineering principles, including coursework in software design, data structures, algorithms, and databases",
    "Emerging Fields: Advanced courses and electives cover modern technologies like Artificial Intelligence, Cloud Computing, Big Data, and Mobile App Development.",
    "Hands-on Labs: The department is equipped with modern computer laboratories and a networked environment designed for continuous practical learning.",
    "Industry Readiness: Graduates are highly sought after by local tech companies, financial institutions, and software houses."
]

vector_store = QdrantVectorStore.from_texts(
    texts = data,
    embedding = embeddingmodel,
    client = qdrant_client,
    collection_name=collection_name
)

retriever = vector_store.as_retriever(search_kwargs = {"k" : 1})


@tool
def RAGsearch(query : str):
  print("Running RAG Model!!")
  retrieved_documents = retriever.invoke(query)
  if not retrieved_documents:
    return "Retriever was unable to get any thing."
  return retrieved_documents

@tool
def calculate(expression : str):
  clean_expression = re.sub(r'[^0-9+\-*/().\s]')
  try:
    math_result = eval(clean_expression)
    return f"mathematical output result : {math_result}"
  except Exception as ex:
    return f"Could not compose calculation due to Error : {str(ex)}"

@tool
def web_search(query : str):
  if "salary" in query:
    return "The salary of a Senior Software Engineer is between 80k to 300k"
  return "The document has been satisfied!!!"


all_tools = ["RAGsearch", "calculate", "web_search"]

print("Initializing the LLM")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
llmwithTools = llm.bind_tools(all_tools)


def autonomous_agent(prompt : str):
  prompt_template = ChatPromptTemplate.from_messages(
      [
          ("System", "You are a elite academic Operations Advisor."),
          ("User", "{input}")
      ]
  )

  promptmessage = prompt_template.format_messages(input=prompt)

  model_decision = llmwithTools.invoke(promptmessage)

  if model_decision.tool_calls:

    for tool_call in model_decision:
      decision_name = tool_call["name"]
      toolarguments = tool_call["args"]

      if decision_name == "RAGsearch": # RAGsearch()
        tool_feedback = RAGsearch.invoke(toolarguments)
      elif decision_name == "calculate":
        tool_feedback = calculate.invoke(toolarguments)
      elif decision_name == "web_search":
        tool_feedback = web_search.invoke(toolarguments)
      else:
        tool_feedback = f"An Error Occurred."


    synthesis_prompt = ChatPromptTemplate.from_messages(
        [
            ("System", f"Synthesize a highly accurate responses back into the model to construct the file. Use the feedback {tool_feedback}"),
            ("user", "{input}"),
            ("ai", "Processing specialized tooling ....")
        ]
    )

    finalOutput = llm.invoke(synthesis_prompt.format_messages(input=prompt))
    print(finalOutput.content)


autonomous_agent()
