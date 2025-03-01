from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import os
# Set OpenAI API Key
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGSMITH_API_KEY"]= os.getenv("LANGSMITH_API_KEY")

from src.retrival import Retrieval

class Generate():
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
        self.retrival= Retrieval()
        self.store = {}
    
    def get_session_history(self,session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]


    def simple_generation(self,query):
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        Always say "thanks for asking!" at the end of the answer.
        {context}
        Question: {query}
        Helpful Answer:"""

        custom_rag_prompt = PromptTemplate.from_template(template)
        context=self.retrival.simple_retrival(query)
        rag_chain = custom_rag_prompt | self.llm | StrOutputParser()
        response = rag_chain.invoke({"query":query,"context":self.format_docs(context)})
        return context, response
    
    def format_docs(self,docs):
        return "\n\n".join(chunk for chunk in docs["document"])
    
    def contextual_generation(self,query):

        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        retrival = self.retrival.vectorstore.as_retriever()
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retrival, contextualize_q_prompt
        )

        
        system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""
        rag_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

        #context=self.retrival.simple_retrival(query)
        question_answer_chain = create_stuff_documents_chain(self.llm, rag_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
            )
        response = conversational_rag_chain.invoke({"input":query},config={"configurable": {"session_id": "abc123"}})
        return {"context":"abc"}, response["answer"]