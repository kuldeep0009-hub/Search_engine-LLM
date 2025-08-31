# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
# from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchResults
# from langchain.agents import initialize_agent,AgentType
# from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# ## isse basically jab hum search kareneg to poora thinking process show hota hai nh ki directly result dikhta h sirf

# import os
# from dotenv import load_dotenv

# #user will input api key so need of entering load_dotenv() bcz 
# # we will not load key from .env user will inout that


# ## Arxiv and wikipedia Tools
# arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200) 
# arxiv=ArxivQueryRun (api_wrapper=arxiv_wrapper)

# api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
# wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

# search=DuckDuckGoSearchResults(name="search")

# st.title("Langchain -chat with search")
# '''
# In this example, we're using StreamlitCallbackHandler. to display the thoughts and actions of an agent in an interactive Streamlit app.
# Try more LangChain
# Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent]'''


# ##Sidebar• for settings
# st. sidebar.title("Settings")
# api_key=st.sidebar.text_input ("Enter your Groq AI API Key:", type="password")

# if "messages" not in st.session_state:
#     st.session_state["messages"]=[
#         {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you"}
#     ]

# for msg in st.session_state.messages:  #ye message schat history dega role me user ya assistant dalo phir content me history dega
#     st.chat_message(msg["role"]).write(msg['content'])


# if prompt:=st.chat_input(placeholder="What is machine learning?"): ##agar user kuch bhi likhega to vo prompt me save ho jaega
#     st.session_state.messages.append({"role":"user","content":prompt})
#     st.chat_message("user"). write(prompt)

#     llm=ChatGroq(groq_api_key=api_key,model_name="llama3-8b-8192",streaming=True)
#     #streaming true se jaise jaise word generate hota rhega saamne aata rhega nh ki poora output generate backend me hokr phir ek baar me saamne aaye 
#     tools=[search,arxiv,wiki]

#     search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True,verbose=True)

#     with st.chat_message("assistant"):
#         st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
#         response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        
#         st.session_state.messages.append({'role': 'assistant', "content": response})
#         st.write(response)
import streamlit as st
from langchain.chat_models import HuggingFaceHub
from langchain.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.streamlit import StreamlitCallbackHandler

st.title("LangChain Chatbot with Search")

# Sidebar for API key
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your HuggingFace API Key:", type="password")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm your chatbot. How can I help you today?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Initialize search tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

duck_search = DuckDuckGoSearchResults(name="search")

tools = [arxiv_tool, wiki_tool, duck_search]

# Handle user input
if prompt := st.chat_input(placeholder="Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize HuggingFace LLM
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct",
                         model_kwargs={"temperature": 0, "max_length": 512},
                         huggingfacehub_api_token=api_key)

    # Initialize agent
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                    handle_parsing_errors=True, verbose=True)

    # Run agent with Streamlit callback
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
