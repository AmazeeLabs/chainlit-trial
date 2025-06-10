import os
import numpy as np
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, cast

import chainlit as cl
from pydantic import SecretStr
from typing import cast

embeddings = OpenAIEmbeddings(
    model="embeddings",  # use env variable
    base_url=os.environ.get("AI_LLM_API_URL"),
    api_key=cast(SecretStr, os.environ.get("AI_LLM_API_TOKEN")),
)


class InMemoryFileStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="embeddings",
            base_url=os.environ.get("AI_LLM_API_URL"),
            api_key=cast(SecretStr, os.environ.get("AI_LLM_API_TOKEN")),
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.file_contents: Dict[str, List[Dict[str, Any]]] = {}

    async def process_file(self, file: cl.File) -> None:
        """Process a file and store its embeddings in memory"""
        try:
            if not file.path:
                return
            with open(file.path, "rb") as f:
                content = f.read()

            text = content.decode("utf-8")
            chunks = self.text_splitter.split_text(text)
            embeddings = await self.embeddings.aembed_documents(chunks)

            self.file_contents[file.id] = [
                {"content": chunk, "embedding": embedding}
                for chunk, embedding in zip(chunks, embeddings)
            ]
        except Exception as e:
            print(f"Error processing file {file.name}: {str(e)}")
            raise

    def search_similar(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar content across all stored files"""
        all_chunks = []
        for file_chunks in self.file_contents.values():
            all_chunks.extend(file_chunks)

        if not all_chunks:
            return []

        # Calculate cosine similarity
        similarities = []
        for chunk in all_chunks:
            similarity = np.dot(query_embedding, chunk["embedding"]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk["embedding"])
            )
            similarities.append((similarity, chunk))

        # Sort by similarity and return top results
        similarities.sort(reverse=True)
        return [chunk for _, chunk in similarities[:limit]]


# Initialize in-memory file store
file_store = InMemoryFileStore()


@cl.on_chat_start
async def on_chat_start():
    memory = MemorySaver()

    graph = create_react_agent(
        ChatOpenAI(
            model="claude-3-5-sonnet",
            temperature=0,
            base_url=os.environ.get("AI_LLM_API_URL"),
            api_key=cast(SecretStr, os.environ.get("AI_LLM_API_TOKEN")),
        ),
        [],
        checkpointer=memory,
    )

    cl.user_session.set("graph", graph)


@cl.on_message
async def on_message(msg: cl.Message):
    # Process attached files if any
    if msg.elements:
        for element in msg.elements:
            if isinstance(element, cl.File):
                await file_store.process_file(element)

    graph = cast(CompiledStateGraph, cl.user_session.get("graph"))
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    full_response = ""

    # Generate embedding for the query
    query_embedding = embeddings.embed_query(msg.content)

    # Get similar documents from both vector DB and file store
    file_similar_docs = file_store.search_similar(query_embedding, limit=5)

    # Combine and deduplicate results
    all_docs = file_similar_docs
    seen_contents = set()
    unique_docs = []
    for doc in all_docs:
        if doc["content"] not in seen_contents:
            seen_contents.add(doc["content"])
            unique_docs.append(doc)

    # Create context from similar documents
    context = "\n\n".join(
        [doc["content"] for doc in unique_docs[:5]]
    )  # Limit to top 5 results

    # Create enhanced prompt with context
    enhanced_prompt = f"""
Context: {context}
Request: {msg.content}
"""

    async for m, _ in graph.astream(
        {"messages": [HumanMessage(content=enhanced_prompt)]},
        stream_mode="messages",
        config=RunnableConfig(
            callbacks=[cb] if os.environ.get("DEBUG") else [],
            configurable={"thread_id": cl.context.session.id},
        ),
    ):
        if isinstance(m, AIMessageChunk) and m.content:
            content = cast(str, cast(AIMessageChunk, m).content)
            full_response += content
            await final_answer.stream_token(content)

    await final_answer.send()


@cl.action_callback("follow_up")
async def on_follow_up(action: cl.Action):
    question = action.payload.get("question", "")
    msg = cl.Message(content=question)
    await on_message(msg)
