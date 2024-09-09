# LangChain: Chat with Search

This Streamlit application allows you to interact with a chatbot that can search the web using the LangChain library and the Groq API. The chatbot can utilize various tools such as Arxiv, Wikipedia, and DuckDuckGo for searching and providing information.

## Screenshot
![Screenshot](https://github.com/aadhil96/LangChain-Chat-with-Search/blob/a624e5eaced9788770628b696d7eabfef3291491/screenshot.JPG)

## Features

- Interactive chat interface with Streamlit.
- Integration with the Groq API for language model processing.
- Search capabilities using Arxiv, Wikipedia, and DuckDuckGo.
- Display of agent thoughts and actions using `StreamlitCallbackHandler`.

## Prerequisites

Before running the application, ensure you have the following prerequisites:

- Python 3.7 or higher.
- Streamlit installed.
- LangChain library installed.
- Validators library installed.
- Groq API key.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. Install the required Python packages:

    ```bash
    pip install streamlit langchain langchain-groq langchain-community dotenv
    ```

## Usage

1. Run the Streamlit application:

    ```bash
    streamlit run your_script_name.py
    ```

2. Open the provided URL in your web browser.

3. Enter your Groq API key in the sidebar.

4. Start chatting with the bot by entering your queries in the chat input field.

5. The chatbot will respond with information gathered from Arxiv, Wikipedia, and DuckDuckGo searches.

## Code Overview

The main components of the code are as follows:

- **Streamlit Configuration**: Sets up the Streamlit page configuration and title.
- **Input Fields**: Collects the Groq API key from the sidebar.
- **API Wrappers and Tools**: Initializes the Arxiv, Wikipedia, and DuckDuckGo search tools.
- **Chat Interface**: Manages the chat session state and displays messages.
- **Agent Initialization**: Initializes the agent using the Groq API and the search tools.
- **Callback Handler**: Uses `StreamlitCallbackHandler` to display the agent's thoughts and actions.

### Example Code Snippet

```python
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

## Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name="Search")

st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({'role': 'assistant', "content": response})
        st.write(response)
