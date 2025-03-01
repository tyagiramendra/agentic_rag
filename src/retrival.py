from FlagEmbedding import FlagReranker
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGSMITH_API_KEY"]= os.getenv("LANGSMITH_API_KEY")

class Retrieval():
    def __init__(self):
        self.reranker_model = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
        self.vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory="./chroma_db",  
            )
    
    def rerank(self,query, search_results):
        search_results = pd.DataFrame(search_results)
        search_results["scores"] = self.reranker_model.compute_score([[query,chunk] for chunk in search_results["document"]]) 
        return search_results[search_results["scores"]>0]
    
    def simple_retrival(self,query,n_docs=10,s_threshold=0.5):
        retriever=self.vectorstore.as_retriever(search_type="similarity_score_threshold",search_kwargs={"k": n_docs,"score_threshold":s_threshold})
        context=retriever.invoke(query)
        print(f"Retrived Documents:{len(context)}")
        all_docs= []
        if len(context)>0:
            for doc in context:
                doc= {"document":doc.page_content}
                all_docs.append(doc)
        else:
            print("No Results founds.")
        return self.rerank(query,all_docs)   