import os
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Transcripter API",
    description="API for transcribing YouTube videos and answering questions about their content",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set API keys
os.environ["COHERE_API_KEY"] = "8bVZxyEDWgX8qHh9wiIkqegiUBtU3IW7u9hYOF2k"
os.environ["OPENROUTER_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]

# Global variables for storing processed data
vector_store = None
retriever = None
chain = None
current_video_id = None
chat_history = []
user_api_keys = {
    "cohere_api_key": None,
    "openrouter_api_key": None
}

# Pydantic models for request/response

class ApiKeysRequest(BaseModel):
    cohere_api_key: str
    openrouter_api_key: str

class ApiKeysResponse(BaseModel):
    success: bool
    message: str

class VideoIdRequest(BaseModel):
    video_id: str

class VideoIdResponse(BaseModel):
    video_id: str
    success: bool
    message: str

class ChatMessageRequest(BaseModel):
    message: str

class ChatMessageResponse(BaseModel):
    message: str
    response: str
    timestamp: str
    success: bool

class ChatHistoryResponse(BaseModel):
    history: List[Dict[str, str]]
    success: bool

class InitializeChatResponse(BaseModel):
    success: bool
    message: str
    video_id: str
    ready: bool

# Initialize components
def initialize_components():
    """Initialize the LLM and prompt components"""
    global llm, prompt
    
    prompt = PromptTemplate(
        template=(
            "You are an enthusiastic assistant who loves sharing knowledge! "
            "Answer the question based only on the provided context with excitement and positivity. "
            "If the answer isn't in the context, cheerfully say you don't know but encourage the user to ask another question.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Enthusiastic Answer:"
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
    """Initialize components with user-provided API keys"""
    global llm, prompt, user_api_keys
    
    if not user_api_keys["cohere_api_key"] or not user_api_keys["openrouter_api_key"]:
        raise ValueError("API keys not set")
    
    # Set environment variables with user keys
    os.environ["COHERE_API_KEY"] = user_api_keys["cohere_api_key"]
    os.environ["OPENROUTER_API_KEY"] = user_api_keys["openrouter_api_key"]
    os.environ["OPENAI_API_KEY"] = user_api_keys["openrouter_api_key"]
    
    prompt = PromptTemplate(
        template=(
            "You are an enthusiastic assistant who loves sharing knowledge! "
            "Answer the question based only on the provided context with excitement and positivity. "
            "If the answer isn't in the context, cheerfully say you don't know but encourage the user to ask another question.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Enthusiastic Answer:"
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
    """Format retrieved documents for the prompt"""
    return "\n\n".join(d.page_content for d in retrieved_docs)[:4000]

# Initialize components on startup
initialize_components()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "YouTube Transcripter API",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Health check endpoint",
            "GET /transcript/{video_id}": "Get video transcript",
            "POST /extension/set-api-keys": "Set API keys for extension",
            "POST /extension/set-video-id": "Set video ID for extension", 
            "POST /extension/initialize-chat": "Initialize chat session",
            "POST /extension/chat": "Send chat message",
            "GET /extension/chat-history": "Get chat history",
            "GET /extension/status": "Get extension status",
            "DELETE /extension/clear-chat": "Clear chat session"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.get("/transcript/{video_id}")
async def get_transcript(video_id: str):
    """Get YouTube video transcript by video ID"""
    try:
        logger.info(f"Fetching transcript for video ID: {video_id}")
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        
        # Format transcript as plain text
        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript)
        
        return {
            "video_id": video_id,
            "transcript": transcript_text,
            "success": True,
            "message": "Transcript fetched successfully"
        }
    except Exception as e:
        logger.error(f"Error fetching transcript for video {video_id}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch transcript: {str(e)}"
        )

# ===== EXTENSION WORKFLOW ENDPOINTS =====

@app.post("/extension/set-api-keys")
async def set_api_keys(request: ApiKeysRequest):
    """Set API keys for the extension (Cohere and OpenRouter)"""
    global user_api_keys
    
    try:
        user_api_keys["cohere_api_key"] = request.cohere_api_key
        user_api_keys["openrouter_api_key"] = request.openrouter_api_key
        
        # Test the keys by initializing components
        initialize_components_with_user_keys()
        
        return ApiKeysResponse(
            success=True,
            message="API keys set successfully and validated"
        )
    except Exception as e:
        logger.error(f"Error setting API keys: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to set API keys: {str(e)}"
        )

@app.post("/extension/set-video-id")
async def set_video_id(request: VideoIdRequest):
    """Set video ID for the extension"""
    global current_video_id
    
    try:
        # Validate video ID by trying to fetch transcript
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(request.video_id)
        
        current_video_id = request.video_id
        
        return VideoIdResponse(
            video_id=request.video_id,
            success=True,
            message="Video ID set successfully"
        )
    except Exception as e:
        logger.error(f"Error setting video ID: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to set video ID: {str(e)}"
        )

@app.post("/extension/initialize-chat")
async def initialize_chat():
    """Initialize chat session - process video and prepare for chatting"""
    global vector_store, retriever, chain, current_video_id, chat_history, user_api_keys
    
    try:
        if not current_video_id:
            raise HTTPException(
                status_code=400,
                detail="No video ID set. Please set video ID first."
            )
        
        if not user_api_keys["cohere_api_key"] or not user_api_keys["openrouter_api_key"]:
            raise HTTPException(
                status_code=400,
                detail="API keys not set. Please set API keys first."
            )
        
        # Initialize components with user keys
        initialize_components_with_user_keys()
        
        # Get transcript
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(current_video_id)
        transcript_text = " ".join(chunk.text for chunk in transcript.snippets)
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        docs = splitter.create_documents([transcript_text])
        
        # Create embeddings and vector store
        embeddings = CohereEmbeddings(model="embed-english-v3.0")
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        
        # Create the chain
        parallel = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        })
        
        chain = parallel | prompt | llm | StrOutputParser()
        
        # Clear previous chat history
        chat_history = []
        
        logger.info(f"Chat initialized for video: {current_video_id}")
        
        return InitializeChatResponse(
            success=True,
            message="Chat initialized successfully! You can now start chatting.",
            video_id=current_video_id,
            ready=True
        )
        
    except Exception as e:
        logger.error(f"Error initializing chat: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to initialize chat: {str(e)}"
        )

@app.post("/extension/chat")
async def send_chat_message(request: ChatMessageRequest):
    """Send a chat message and get response"""
    global chain, chat_history, current_video_id
    from datetime import datetime
    
    if chain is None:
        raise HTTPException(
            status_code=400,
            detail="Chat not initialized. Please initialize chat first."
        )
    
    try:
        logger.info(f"Processing chat message: {request.message}")
        
        # Get response from the chain
        response = chain.invoke(request.message)
        
        # Add to chat history
        timestamp = datetime.now().isoformat()
        chat_entry = {
            "user_message": request.message,
            "bot_response": response,
            "timestamp": timestamp
        }
        chat_history.append(chat_entry)
        
        return ChatMessageResponse(
            message=request.message,
            response=response,
            timestamp=timestamp,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat message: {str(e)}"
        )

@app.get("/extension/chat-history")
async def get_chat_history():
    """Get chat history"""
    global chat_history
    
    return ChatHistoryResponse(
        history=chat_history,
        success=True
    )

@app.delete("/extension/clear-chat")
async def clear_chat():
    """Clear chat history and reset chat session"""
    global chat_history, vector_store, retriever, chain, current_video_id
    
    chat_history = []
    vector_store = None
    retriever = None
    chain = None
    current_video_id = None
    
    return {
        "success": True,
        "message": "Chat cleared successfully"
    }

@app.get("/extension/status")
async def get_extension_status():
    """Get current extension status"""
    global user_api_keys, current_video_id, chain, chat_history
    
    return {
        "api_keys_set": bool(user_api_keys["cohere_api_key"] and user_api_keys["openrouter_api_key"]),
        "video_id_set": bool(current_video_id),
        "chat_initialized": bool(chain),
        "current_video_id": current_video_id,
        "chat_messages_count": len(chat_history),
        "ready_to_chat": bool(chain and current_video_id)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
