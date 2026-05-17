from typing import TypedDict
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=os.environ.get("GROQ_API_KEY")
)

embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

class AgentState(TypedDict):
    question: str
    retrieved_info: str
    reasoning: str
    final_answer: str

def retrieval_agent(state: AgentState):
    docs = vectorstore.similarity_search(state["question"], k=3)
    retrieved = "\n\n".join([doc.page_content for doc in docs])
    print("[Retriever Agent Finished]")
    return {"retrieved_info": retrieved}

def reasoning_agent(state: AgentState):
    prompt = f"""
    You are a loan risk analyst.
    Question: {state["question"]}
    Relevant information: {state["retrieved_info"]}
    Analyze the situation clearly.
    """
    response = llm.invoke(prompt)
    print("[Reasoning Agent Finished]")
    return {"reasoning": response.content}

def decision_agent(state: AgentState):
    prompt = f"""
    You are a loan decision officer.
    Question: {state["question"]}
    Analysis: {state["reasoning"]}
    - If general question, answer helpfully.
    - If applicant data provided, give APPROVE or REJECT with reason.
    """
    response = llm.invoke(prompt)
    print("[Decision Agent Finished]")
    return {"final_answer": response.content}

graph = StateGraph(AgentState)
graph.add_node("retriever", retrieval_agent)
graph.add_node("reasoner", reasoning_agent)
graph.add_node("decision", decision_agent)
graph.set_entry_point("retriever")
graph.add_edge("retriever", "reasoner")
graph.add_edge("reasoner", "decision")
graph.add_edge("decision", END)

agent_app = graph.compile()