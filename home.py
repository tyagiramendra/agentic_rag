import streamlit as st
import time
from src.graphs.agentic_rag import AgenticRag
from src.graphs.adaptive_rag import AdaptiveRAG
adapt_rag = AdaptiveRAG()

# Streamed response emulator
def response_generator(prompt):
    #contexts,response = generation.contextual_generation(prompt)
    #response,source=adapt_rag._execute(prompt)
    response=adapt_rag._execute(prompt)
    st.write(f"Response:{response}")
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
    #st.write("Source:",source)

st.title("Indian Budget 2025-26 QnA")
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})