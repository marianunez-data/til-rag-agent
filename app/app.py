"""
app.py
------
Streamlit chat interface for the TIL Agent.
"""

from __future__ import annotations

import asyncio

import ingest
import logs
import streamlit as st
from agent import build_agent
from search import SearchPipeline

st.set_page_config(
    page_title="TIL Agent",
    page_icon="",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def initialise_pipeline():
    with st.spinner("Loading TIL repository and building search indexes..."):
        documents = ingest.read_repo_data("jbranchaud", "til", branch="master")
        chunks = ingest.chunk_documents(documents, strategy="sliding_window")
        chunks = ingest.filter_documents(
            chunks,
            exclude_filenames=["README", "CONTRIBUTING"],
            min_content_length=100,
        )
        pipeline = SearchPipeline.build(chunks)
        agent = build_agent(pipeline)
    return agent


agent = initialise_pipeline()

st.title("TIL Agent")
st.caption(
    "Ask technical programming questions about git, vim, PostgreSQL, "
    "JavaScript, Python, Linux, and more. "
    "Grounded in [jbranchaud/til](https://github.com/jbranchaud/til) — "
    "1,700+ curated programming tips."
)
st.divider()
st.info(
    "This agent answers programming questions only. "
    "Examples: *How do I rebase commits in git?* — "
    "*How do I find text across files in vim?* — "
    "*How do I check open ports on Linux?*"
)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


async def run_agent(prompt: str) -> str:
    result = await agent.run(user_prompt=prompt)
    logs.save_interaction(agent, result.new_messages(), source="user")
    return result.output


if prompt := st.chat_input("Ask a programming question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching TIL repository..."):
            response = asyncio.run(run_agent(prompt))
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
