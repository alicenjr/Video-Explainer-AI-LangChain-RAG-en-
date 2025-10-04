# 🎥 Video Explainer AI - YouTube Transcript Q&A System

An intelligent YouTube video transcript analyzer powered by LangChain, RAG (Retrieval-Augmented Generation), and LLM. This FastAPI-based application allows users to ask questions about YouTube video content and get accurate, context-aware answers.

## ✨ Features

- **YouTube Transcript Extraction**: Automatically fetches and processes YouTube video transcripts
- **RAG Implementation**: Uses Retrieval-Augmented Generation for accurate question answering
- **Vector Search**: FAISS-based similarity search for relevant content retrieval
- **Chrome Extension Support**: Complete API endpoints for browser extension integration
- **Chat History**: Maintains conversation context throughout the session
- **Multi-API Support**: Integration with Cohere (embeddings) and OpenRouter (LLM)

## 🏗️ Architecture

The system consists of three main components:

1. **Transcript Processing Pipeline**
   - YouTube transcript extraction using `youtube-transcript-api`
   - Text chunking with RecursiveCharacterTextSplitter
   - Embedding generation using Cohere's embed-english-v3.0

2. **Vector Store & Retrieval**
   - FAISS vector database for efficient similarity search
   - Configurable retriever with k=3 most relevant chunks
   - Context formatting for optimal prompt construction

3. **LLM Chain**
   - LangChain RunnableParallel for efficient processing
   - Meta's Llama 3.2 3B model via OpenRouter
   - Custom prompt engineering for enthusiastic, helpful responses

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Cohere API key
- OpenRouter API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/alicenjr/Video-Explainer-AI-LangChain-RAG-en-.git
cd Video-Explainer-AI-LangChain-RAG-en-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys (you can either hardcode them or use the `/extension/set-api-keys` endpoint)

4. Run the application:
```bash
python api.py
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Core Endpoints

#### Get Transcript
```http
GET /transcript/{video_id}
```
Fetches the transcript for a given YouTube video ID.

#### Health Check
```http
GET /health
```
Returns the API health status.

### Extension Workflow Endpoints

#### 1. Set API Keys
```http
POST /extension/set-api-keys
Content-Type: application/json

{
  "cohere_api_key": "your_cohere_key",
  "openrouter_api_key": "your_openrouter_key"
}
```

#### 2. Set Video ID
```http
POST /extension/set-video-id
Content-Type: application/json

{
  "video_id": "dQw4w9WgXcQ"
}
```

#### 3. Initialize Chat
```http
POST /extension/initialize-chat
```
Processes the video transcript and sets up the RAG chain.

#### 4. Send Chat Message
```http
POST /extension/chat
Content-Type: application/json

{
  "message": "What is this video about?"
}
```

#### 5. Get Chat History
```http
GET /extension/chat-history
```

#### 6. Clear Chat
```http
DELETE /extension/clear-chat
```

#### 7. Get Status
```http
GET /extension/status
```

## 🔧 Technical Stack

- **FastAPI**: High-performance async web framework
- **LangChain**: LLM orchestration and RAG implementation
- **FAISS**: Vector similarity search
- **Cohere**: Text embedding generation
- **OpenRouter**: LLM API gateway (Llama 3.2 3B)
- **YouTube Transcript API**: Video transcript extraction

## 📁 Project Structure

```
├── api.py              # Main FastAPI application
├── runnables.py        # LangChain runnables configuration
├── requirements.txt    # Python dependencies
└── extension/          # Chrome extension files (if applicable)
```

## 🎯 Use Cases

- Educational video content analysis
- Video content summarization
- Research and note-taking from video lectures
- Quick information retrieval from long videos
- Accessibility tool for video content

## 🔐 Security Notes

- API keys are handled securely through environment variables
- CORS is enabled for browser extension integration
- Input validation on all endpoints

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is open-source and available for educational and commercial use.

## 👨‍💻 Author

**alicenjr** - [GitHub Profile](https://github.com/alicenjr)

---

⭐ Star this repo if you find it useful!
