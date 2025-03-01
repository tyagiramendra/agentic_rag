import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pandas as pd
import json
from datasets import Dataset
from ragas.evaluation import evaluate
from ragas.metrics import ( 
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

from src.generation import Generate

# Set OpenAI API Key
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class Ingestion():
    def __init__(self):
        self.embddings = OpenAIEmbeddings(model="text-embedding-3-small")

    def ingest_pipeline(self):
        # 1. Load PDF document
        pdf_loader = PyPDFLoader("budget_speech.pdf")
        documents = pdf_loader.load()

        # 2. Chunk the document
        # Recursive Chunking 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  
            chunk_overlap=100)
        # Semantic 
        #text_splitter = SentenceTransformersTokenTextSplitter(
        #    chunk_overlap=100
        #)
        chunks = text_splitter.split_documents(documents)

        #3. Initialize ChromaDB with LangChain integration
        Chroma.from_documents(
            documents=chunks,
            embedding=self.embddings,
            persist_directory="./chroma_db", 
            collection_name="rag_collection"
        )
        print("Ingestion has been done!.")

class Evaluation():
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
        self.embddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.generate = Generate()

    def execute(self):
        all_data=[]
        df = pd.read_csv("qa_pairs.csv")
        for index, row in df.iterrows():
            ground_truth = row["ground_truth"]
            question= row["question"]
            docs,answer=self.generate.simple_generation(question)
            row={
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "contexts": [doc for doc in docs["document"]]
            }
            all_data.append(row)        
        test_dataset= Dataset.from_list(all_data)
        evaluation_result = evaluate(
            test_dataset,
            metrics=[
                answer_relevancy,
                faithfulness,
                context_recall,
                context_precision,
            ],
            llm=self.llm,
            embeddings=self.embddings
        )
        return evaluation_result

if __name__ == "__main__":
    ingest=Ingestion()
    ingest.ingest_pipeline()
    #eval_pipeline = Evaluation()
    #print(eval_pipeline.execute())