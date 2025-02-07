from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from document_handler import retriever_tool
tools = [retriever_tool]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    

### Edges ===============================================================================


def grade_documents(state) -> Literal["generate", "ignorance"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]
    
    question = next(
                    (msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), None
                )

    docs = last_message.content
    print(last_message)

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        return "ignorance"
    
    

### Nodes ===============================================================================

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    
    return {"messages": [response]}

def ignorance(_):
    """
    Informs the user that the agent does not have the required information to process the query.

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("--LACK OF INFORMATION--")
    
    response = AIMessage(content="""
                        I'm sorry, but I don't have enough information to accurately process your request. \n
                        Could you provide more details or clarify what you're looking for? \n
                        If possible, consider uploading relevant documents to support your request so I can perform RAG. \n
                        I'm happy to help once I have more information!
                        """)
    
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = next(
                    (msg.content for msg in reversed(messages) if isinstance(msg, HumanMessage)), None
                )
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


### Graph ===============================================================================


workflow = StateGraph(AgentState)

workflow.add_node("agent", agent)  # agent
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("generate", generate )  # Generating a response after we know the documents are relevant
workflow.add_node("ignorance", ignorance) # informs user about lack of information for processing their query

# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)

# informs user about lack of information for processing their query
workflow.add_edge("ignorance",END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


def stream_graph_updates(prompt: str, thread_id: str):
    config = {"configurable": {"thread_id": str(thread_id)}}
    for chunk, _ in graph.stream(
        {"messages": [("user", prompt)]},
        config,
        stream_mode="messages"
    ):
        if isinstance(chunk, AIMessage):
            yield chunk.content