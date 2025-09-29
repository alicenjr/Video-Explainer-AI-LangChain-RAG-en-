import os
import logging
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI
app = FastAPI(
    title="YouTube Transcripter API (BUGGY)",
    description="API with intentional logic errors",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment keys (dummy for testing)
os.environ["COHERE_API_KEY"] = "dummy_cohere_key"
os.environ["OPENROUTER_API_KEY"] = "dummy_openrouter_key"
os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]

# Global state
vector_store = None
retriever = None
chain = None
current_video_id = None
chat_history = []
user_api_keys = {"cohere_api_key": None, "openrouter_api_key": None}

# Pydantic models
class ApiKeysRequest(BaseModel):
    cohere_api_key: str
    openrouter_api_key: str

class VideoIdRequest(BaseModel):
    video_id: str

class ChatMessageRequest(BaseModel):
    message: str

# ===== Initialize LLM and prompt =====
def initialize_components():
    global llm, prompt
    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant.\nContext:\n{context}\nQuestion: {question}\nAnswer:"
        ),
        input_variables=["question", "context"],
    )
    llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct:free",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=512,
    )

def initialize_components_with_user_keys():
    global llm, prompt, user_api_keys
    # LOGIC ERROR: Not checking if keys are valid, will silently fail
    os.environ["COHERE_API_KEY"] = user_api_keys.get("cohere_api_key", "")
    os.environ["OPENROUTER_API_KEY"] = user_api_keys.get("openrouter_api_key", "")
    os.environ["OPENAI_API_KEY"] = user_api_keys.get("openrouter_api_key", "")
    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant.\nContext:\n{context}\nQuestion: {question}\nAnswer:"
        ),
        input_variables=["question", "context"],
    )
    llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct:free",
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=512,
    )

def format_docs(retrieved_docs):
    # LOGIC ERROR: Truncate blindly, may cut useful info
    return "\n\n".join(d.page_content for d in retrieved_docs)[:200]  # too short

initialize_components()

# ===== Endpoints =====
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/extension/set-api-keys")
async def set_api_keys(request: ApiKeysRequest):
    global user_api_keys
    user_api_keys["cohere_api_key"] = request.cohere_api_key
    user_api_keys["openrouter_api_key"] = request.openrouter_api_key
    try:
        initialize_components_with_user_keys()
        return {"success": True, "message": "API keys set"}
    except:
        # LOGIC ERROR: Catch-all hides actual errors
        return {"success": False, "message": "Failed to set API keys"}

@app.post("/extension/set-video-id")
async def set_video_id(request: VideoIdRequest):
    global current_video_id
    current_video_id = request.video_id
    # LOGIC ERROR: Does not validate if video exists
    return {"success": True, "video_id": request.video_id}

@app.post("/extension/initialize-chat")
async def initialize_chat():
    global vector_store, retriever, chain, current_video_id, chat_history
    if not current_video_id:
        raise HTTPException(status_code=400, detail="No video ID set")
    # LOGIC ERROR: This will break if transcript API changes
    transcript = YouTubeTranscriptApi().fetch(current_video_id)
    # LOGIC ERROR: Wrong attribute (snippets does not exist)
    transcript_text = " ".join(chunk.text for chunk in transcript.snippets)  
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([transcript_text])
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    # LOGIC ERROR: Overwriting chain every time; global state shared across users
    chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }) | prompt | llm | StrOutputParser()
    chat_history = []  # resets history
    return {"success": True, "message": "Chat initialized"}

@app.post("/extension/chat")
async def send_chat_message(request: ChatMessageRequest):
    global chain, chat_history
    if chain is None:
        raise HTTPException(status_code=400, detail="Chat not initialized")
    # LOGIC ERROR: If chain fails, exception is uncaught and may leak keys
    response = chain.invoke(request.message)
    from datetime import datetime
    chat_history.append({
        "user_message": request.message,
        "bot_response": response,
        "timestamp": datetime.now().isoformat()
    })
    return {"message": request.message, "response": response, "success": True}

@app.get("/extension/chat-history")
async def get_chat_history():
    global chat_history
    # LOGIC ERROR: Global chat_history shared across all users
    return {"history": chat_history, "success": True}

@app.delete("/extension/clear-chat")
async def clear_chat():
    global chat_history, vector_store, retriever, chain, current_video_id
    # LOGIC ERROR: Clears everything blindly, loses video context
    chat_history = []
    vector_store = None
    retriever = None
    chain = None
    current_video_id = None
    return {"success": True, "message": "Chat cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
