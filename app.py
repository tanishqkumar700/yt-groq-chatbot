import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# 🎯 Sahi aur Up-to-date text splitter import jo aapne bataya
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Models & Embeddings (No GroqEmbeddings bug anymore!)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Load variables safely from the .env file
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="YouTube Chatbot Backend")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_cache = {}

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

class InitializeRequest(BaseModel):
    video_id: str

class ChatRequest(BaseModel):
    video_id: str
    question: str


@app.post("/initialize")
async def initialize_video(data: InitializeRequest):
    video_id = data.video_id
    
    if video_id in video_cache:
        return {"status": "success", "message": "Video already indexed."}
        
    try:
        # # 1. Fetch transcript from YouTube using the new API format
        api_instance = YouTubeTranscriptApi()
        transcript_obj = api_instance.fetch(video_id)
        
        # Flatten the modern block objects into a single plain text string
        transcript = " ".join(block.text for block in transcript_obj)
        
        # # 2. Split text using the modern splitter import you verified
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        
        # # 3. Use standard light-weight embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # # 4. Create FAISS Database
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Cache retriever locally for your chat endpoint
        video_cache[video_id] = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        print(f"✅ Successfully indexed video: {video_id}")
        return {"status": "success", "message": "Video transcript indexed successfully!"}
        
    except TranscriptsDisabled:
        raise HTTPException(status_code=400, detail="Captions are disabled for this YouTube video.")
    except Exception as e:
        print(f"❌ Error during initialization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_with_video(data: ChatRequest):
    video_id = data.video_id
    
    if video_id not in video_cache:
        raise HTTPException(status_code=400, detail="Video not initialized. Call /initialize first.")
        
    try:
        retriever = video_cache[video_id]
        
        # Llama 3.3 70B Model via Groq cloud server
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
        
        prompt = PromptTemplate(
            template="""
              You are a helpful assistant.
              Answer ONLY from the provided transcript context.
              If the context is insufficient, just say you don't know.

              {context}
              Question: {question}
            """,
            input_variables=['context', 'question']
        )
        
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        
        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser
        
        answer = main_chain.invoke(data.question)
        return {"answer": answer}
        
    except Exception as e:
        print(f"❌ Error during chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)