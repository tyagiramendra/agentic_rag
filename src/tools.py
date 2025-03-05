from langchain.tools.retriever import create_retriever_tool
from src.retrival import Retrieval
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun, TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper

retrival=Retrieval()
retriever_tool = create_retriever_tool(
    retrival.vectorstore.as_retriever(),
    name="Indian Budget Search 2025-2026",
    description="Search for information about indian budget 2025-2026. For any question about indian budget 2025-2026. You can user this tool. "
)

api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wikipedia_tool=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv_tool=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
web_search_tool = TavilySearchResults(k=4)