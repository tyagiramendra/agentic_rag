"""
File: adaptive rag.py
Description: Adaptive RAG Query Analysis + (RAG + Self reflection)
Author: Ramendra Tyagi
"""

import os
from dotenv import load_dotenv
load_dotenv() 
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

from langgraph.graph import START, END, StateGraph 

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from typing import Literal

from src.graphs.prompts import question_router, rag_chain, question_rewriter, hallucination_grader, answer_grader
from src.models import AgentState, Grade
from src.tools import web_search_tool, retriever_tool, retrival


class AdaptiveRAG():
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def route_question(self,state):
        """
        Route question to web search or RAG.
        Args:
            state (dict): The current graph state
        Returns:
            str: Next node to call
        """
        print("---ROUTE QUESTION---")
        question = state["question"]
        source = question_router().invoke({"question": question})
        if source.datasource == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif source.datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
    
    def web_search(self,state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]

        # Web search
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)

        return {"documents": web_results, "question": question}
    
    def retrieve(self,state):
        """
        Retrieve documents
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]
        vdb_retrival = retrival.vectorstore.as_retriever()
        # Retrieval
        documents = vdb_retrival.invoke(question)
        return {"documents": documents, "question": question}
    
    def generate(self,state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = rag_chain().invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}
    
    def grade_documents(self,state) -> Literal["generate", "transform_query"]:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        print("---CHECK RELEVANCE---")
        # LLM with tool and validation
        llm_with_tool = self.model.with_structured_output(Grade)

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
        documents = state["documents"]
        question = state["question"]

        scored_result = chain.invoke({"question": question, "context": documents})

        score = scored_result.binary_score

        if score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "generate"
        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            print(score)
            return "transform_query"
        
    def transform_query(self,state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = question_rewriter().invoke({"question": question})
        return {"documents": documents, "question": better_question}
    
    def grade_generation_v_documents_and_question(self,state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = hallucination_grader().invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader().invoke({"question": question, "generation": generation})
            grade = score.binary_score
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"


    def create_workflow(self):
        workflow= StateGraph(AgentState)
        # Define Nodes
        workflow.add_node("websearch",self.web_search)
        workflow.add_node("retrieve",self.retrieve)
        workflow.add_node("grade_documents",self.grade_documents)
        workflow.add_node("generate",self.generate)
        workflow.add_node("transform_query", self.transform_query)


        #create workflow
        workflow.add_conditional_edges(
            START,self.route_question,
            {
                "web_search":"websearch",
                "vectorstore":"retrieve"
            }
        )
        workflow.add_edge("websearch","generate")
        workflow.add_conditional_edges(
            "retrieve",
            self.grade_documents,
        )
        workflow.add_edge("transform_query","retrieve")
        workflow.add_edge("retrieve","generate")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )
        self.adaptive_agent= workflow.compile()
    
    def _execute(self, query):
        self.create_workflow()
        response = self.adaptive_agent.invoke({"question":query})
        return response["generation"]

