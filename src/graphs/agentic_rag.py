import os
from dotenv import load_dotenv
load_dotenv() 
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

from typing import Annotated, Literal, Sequence
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# langgraph 
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from langchain_groq import ChatGroq
from src.tools import retriever_tool, wikipedia_tool
from src.models import Grade, AgentState

tools=[retriever_tool,wikipedia_tool]

class AgenticRag():
    def __init__(self):
        self.model = ChatGroq(model="qwen-2.5-32b")
    
    def agent(self,state):
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
        model = ChatGroq(model="qwen-2.5-32b")
        model = model.bind_tools(tools)
        response = model.invoke(messages)
        return {"messages": [response]}

    def generate(self,state):
        """
        Generate answer
        Args:
            state (messages): The current state
        Returns:
            dict: The updated message
        """
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]

        docs = last_message.content

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # Chain
        rag_chain = prompt | self.model | StrOutputParser()

        # Run
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}
    
    # Post-processing
    def format_docs(self,docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    
    def rewrite(self,state):
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """

        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
            )
        ]

        response = self.model.invoke(msg)
        return {"messages": [response]}
    
    def grade_documents(self,state) -> Literal["generate", "rewrite"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        print("---CHECK RELEVANCE---")
        # LLM
        model = ChatGroq(model="qwen-2.5-32b")

        # LLM with tool and validation
        llm_with_tool = model.with_structured_output(Grade)

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

        question = messages[0].content
        docs = last_message.content

        scored_result = chain.invoke({"question": question, "context": docs})

        score = scored_result.binary_score

        if score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "generate"

        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            print(score)
            return "rewrite"

    def create_workflow(self):
        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the nodes we will cycle between
        workflow.add_node("agent", self.agent)  # agent
        retrieve = ToolNode([retriever_tool,wikipedia_tool])
        workflow.add_node("retrieve", retrieve)  # retrieval
        workflow.add_node("rewrite", self.rewrite)  # Re-writing the question
        workflow.add_node(
            "generate", self.generate
        )  
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
            self.grade_documents,
        )
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")

        # Compile
        self.graph = workflow.compile()

    def _execute(self,query):
        self.create_workflow()
        response=self.graph.invoke({"messages":query})
        source_tool=response["messages"][-2].name
        return response["messages"][-1].content, source_tool

if __name__ == "__main__":
    agentic_rag = AgenticRag()
    agentic_rag._execute("What is machine learning")
    print(agentic_rag)