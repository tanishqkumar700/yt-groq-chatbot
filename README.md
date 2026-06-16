# 🎥 YouTube Contextual Chat Assistant

A lightweight browser extension that lets you chat directly with any YouTube video in real-time. It uses a local RAG pipeline to answer questions instantly and accurately based on the video's subtitles.

## 🧠 System Architecture

```text
[YouTube] ──► [FastAPI Backend] ──► [LangChain (Text Splitting)]
                                                │
                                                ▼
[Groq Cloud] ◄── [FAISS Index] ◄── [HuggingFace Embeddings]
```

1. **Ingestion & Chunking**: Fetches YouTube transcripts and splits them into manageable chunks.
2. **Local Embeddings**: Generates vector embeddings locally using HuggingFace.
3. **FAISS Indexing**: Stores vectors in an in-memory FAISS database for fast retrieval.
4. **Inference**: Queries Groq Cloud (Llama 3.3) with relevant context to generate precise answers.

---

## 📸 Screenshots

| Extension UI | Chat Interaction |
| :---: | :---: |
| ![UI Screenshot](screenshots/Image2.png) | ![Chat Screenshot]![YouTube Assistant Extension](screenshots/image1.png) |

---

## 🚀 Installation & Usage

### 1. Set Up the Backend
Requires **Python 3.8+** and a [Groq API Key](https://console.groq.com/).

```bash
git clone https://github.com/tanishqkumar700/yt-groq-chatbot.git
cd yt-groq-chatbot

# Create virtual environment
python -m venv venv
# Activate: `.\venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)

# Install dependencies using requirements.txt
pip install -r requirements.txt
```

Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

Start the server:
```bash
python app.py
```
*(Server runs on `http://127.0.0.1:8000`)*

### 2. Install the Browser Extension
1. Open your Chromium browser (Chrome/Edge/Brave) and go to `chrome://extensions/`.
2. Enable **Developer mode** (top right).
3. Click **Load unpacked** and select the `yt-groq-chatbot` project folder.

### 3. Start Chatting!
1. Go to any YouTube video with closed captions.
2. Click the extension icon in your toolbar.
3. Wait for the **"✨ Ready to Chat!"** status, then type your questions!