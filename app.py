import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
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
# from dotenv import load_dotenv
# load_dotenv()

from config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_K,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(title="YouTube Chatbot Backend")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "youtube-chatbot-backend"
    }


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
    video_id: str = Field(
        ...,
        min_length=11,
        max_length=11,
        description="YouTube video ID"
    )

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, value):
        allowed_characters = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789-_"
        )

        if not all(char in allowed_characters for char in value):
            raise ValueError("Invalid YouTube video ID format")

        return value


class ChatRequest(BaseModel):
    video_id: str = Field(
        ...,
        min_length=11,
        max_length=11,
        description="YouTube video ID"
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question about the video"
    )

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, value):
        allowed_characters = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789-_"
        )

        if not all(char in allowed_characters for char in value):
            raise ValueError("Invalid YouTube video ID format")

        return value


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
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.create_documents([transcript])
        
        # # 3. Use standard light-weight embeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # # 4. Create FAISS Database
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Cache retriever locally for your chat endpoint
        video_cache[video_id] = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVAL_K})
        logger.info("Successfully indexed video: %s", video_id)
        return {"status": "success", "message": "Video transcript indexed successfully!"}
        
    except TranscriptsDisabled:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "CAPTIONS_DISABLED",
                "message": "Captions are disabled for this YouTube video."
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "Error during initialization for video %s: %s",
            video_id,
            str(e)
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INITIALIZATION_FAILED",
                "message": "Unable to process this YouTube video."
            }
        )


@app.post("/chat")
async def chat_with_video(data: ChatRequest):
    video_id = data.video_id
    
    if video_id not in video_cache:
        raise HTTPException(status_code=400, detail="Video not initialized. Call /initialize first.")
        
    try:
        retriever = video_cache[video_id]
        
        # Llama 3.3 70B Model via Groq cloud server
        llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        
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

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "Error during chat for video %s: %s",
            video_id,
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail={
                "code": "CHAT_FAILED",
                "message": "Unable to generate an answer for this question."
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)