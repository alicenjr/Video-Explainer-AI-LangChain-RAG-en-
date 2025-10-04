# 🎥 Video Explainer AI

Ask questions about YouTube videos and get instant answers using AI.

## What It Does

This app extracts YouTube transcripts and uses AI to answer your questions about the video content. It's powered by LangChain, RAG, and Meta's Llama model.

## Features

- Extract YouTube transcripts automatically
- Ask questions in natural language
- Get accurate answers with context
- Chrome extension support
- Maintains chat history

## Tech Stack

**Backend**: FastAPI, LangChain, FAISS  
**AI**: Cohere (embeddings), OpenRouter (Llama 3.2 3B)  
**Other**: YouTube Transcript API

## Quick Start

1. **Install**
```bash
git clone https://github.com/alicenjr/Video-Explainer-AI-LangChain-RAG-en-.git
cd Video-Explainer-AI-LangChain-RAG-en-
pip install -r requirements.txt
```

2. **Run**
```bash
python api.py
```

3. **Access** at `http://localhost:8000`

## How to Use

### Set up API keys
```http
POST /extension/set-api-keys
{
  "cohere_api_key": "your_key",
  "openrouter_api_key": "your_key"
}
```

### Analyze a video
```http
POST /extension/set-video-id
{ "video_id": "dQw4w9WgXcQ" }

POST /extension/initialize-chat
```

### Ask questions
```http
POST /extension/chat
{ "message": "What is this video about?" }
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/transcript/{video_id}` | GET | Get video transcript |
| `/extension/set-api-keys` | POST | Configure API keys |
| `/extension/set-video-id` | POST | Set YouTube video |
| `/extension/initialize-chat` | POST | Process transcript |
| `/extension/chat` | POST | Ask questions |
| `/extension/chat-history` | GET | View chat history |
| `/extension/clear-chat` | DELETE | Clear conversation |
| `/extension/status` | GET | Check system status |

## Use Cases

- Study from educational videos
- Research video content quickly
- Take notes from lectures
- Summarize long videos
- Make video content accessible

## Author

**alicenjr** - [GitHub](https://github.com/alicenjr)

---

⭐ Star this repo if you find it useful!
