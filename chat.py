from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    
graph_builder = StateGraph(State)

llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
        
# Nodes
graph_builder.add_node("chatbot", chatbot)

# Edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Final graph
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def stream_graph_updates(prompt: str, thread_id: str):
    config = {"configurable": {"thread_id": str(thread_id)}}
    for chunk, _ in graph.stream(
        {"messages": [("user", prompt)]},
        config,
        stream_mode="messages"
    ):
        if isinstance(chunk, AIMessage):
            yield chunk.content
        
        
        

