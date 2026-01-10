"""
Chat Service for Streamlit App.
Integrates RAG, STT, and TTS services directly.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Generator
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


# System prompt for grounded responses
SYSTEM_PROMPT = """You are a helpful government services assistant. Your role is to help citizens understand government services and procedures.

CRITICAL INSTRUCTIONS:
1. ONLY answer based on the provided context from official government documents
2. If the context doesn't contain relevant information, say "I don't have information about that in my knowledge base. Please contact the relevant government office directly."
3. NEVER make up or assume information not present in the context
4. Be concise but complete in your answers
5. If asked about fees, processing times, or requirements, quote the exact information from the context
6. For follow-up questions, use the conversation history to maintain context

Context from Government Documents:
{context}

Remember: Accuracy is paramount. Citizens depend on correct information."""


class ChatService:
    """
    Unified chat service that handles:
    - LLM inference (via Ollama)
    - RAG retrieval
    - Model switching
    """
    
    # Available Ollama models optimized for M1 Mac
    AVAILABLE_MODELS = {
        "llama3.2": {"name": "Llama 3.2 (2B)", "description": "Fast, good for simple Q&A", "size": "2B"},
        "llama3.2:3b": {"name": "Llama 3.2 (3B)", "description": "Better quality, still fast", "size": "3B"},
        "mistral": {"name": "Mistral (7B)", "description": "Excellent quality", "size": "7B"},
        "llama3.1:8b": {"name": "Llama 3.1 (8B)", "description": "Best quality, slower", "size": "8B"},
        "phi3": {"name": "Phi-3 (3.8B)", "description": "Microsoft's compact model", "size": "3.8B"},
        "gemma2:2b": {"name": "Gemma 2 (2B)", "description": "Google's efficient model", "size": "2B"},
    }
    
    def __init__(
        self,
        model_name: str = "llama3.2",
        ollama_base_url: str = "http://localhost:11434",
        knowledge_dir: str = "./knowledge",
        chroma_dir: str = "./data/chroma"
    ):
        """Initialize the chat service."""
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.knowledge_dir = Path(knowledge_dir)
        self.chroma_dir = Path(chroma_dir)
        
        self._embeddings = None
        self._vector_store = None
        self._llm = None
        self._prompt = None
        
    def _ensure_initialized(self):
        """Lazy initialization of components."""
        if self._embeddings is None:
            self._init_embeddings()
        if self._vector_store is None:
            self._init_vector_store()
        if self._llm is None:
            self._init_llm()
    
    def _init_embeddings(self):
        """Initialize embedding model."""
        logger.info("Loading embedding model (this may take a moment on first run)...")
        try:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _init_vector_store(self):
        """Initialize vector store."""
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        
        self._vector_store = Chroma(
            collection_name="government_services",
            embedding_function=self._embeddings,
            persist_directory=str(self.chroma_dir)
        )
        
        # Check if we need to ingest documents
        if self._vector_store._collection.count() == 0:
            self._ingest_knowledge_base()
    
    def _init_llm(self):
        """Initialize the LLM."""
        logger.info(f"Initializing LLM: {self.model_name}")
        
        self._llm = ChatOllama(
            model=self.model_name,
            base_url=self.ollama_base_url,
            temperature=0.1,
            num_predict=1024,
        )
        
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        logger.info("LLM initialized")
    
    def _ingest_knowledge_base(self):
        """Ingest knowledge documents into vector store."""
        if not self.knowledge_dir.exists():
            logger.warning(f"Knowledge directory not found: {self.knowledge_dir}")
            return
        
        loader = DirectoryLoader(
            str(self.knowledge_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            return
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        self._vector_store.add_documents(chunks)
        logger.info("Documents ingested")
    
    def switch_model(self, model_name: str):
        """Switch to a different LLM model."""
        if model_name != self.model_name:
            self.model_name = model_name
            self._llm = None  # Force re-initialization
            self._init_llm()
    
    def chat(
        self,
        message: str,
        conversation_history: list[dict] = None,
        stream: bool = False
    ) -> dict:
        """
        Process a chat message and return response.
        
        Args:
            message: User's message
            conversation_history: Previous messages for context
            stream: Whether to stream the response
            
        Returns:
            dict with response, sources, and confidence
        """
        self._ensure_initialized()
        
        conversation_history = conversation_history or []
        
        # Retrieve relevant documents
        retrieval_results = self._vector_store.similarity_search_with_relevance_scores(
            message, k=4
        )
        
        # Build context
        context_parts = []
        sources = []
        
        for doc, score in retrieval_results:
            context_parts.append(doc.page_content)
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": Path(doc.metadata.get("source", "unknown")).name,
                "relevance_score": round(score, 3)
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Format history
        history_messages = []
        for turn in conversation_history[-10:]:  # Last 10 turns
            if turn.get("role") == "user":
                history_messages.append(HumanMessage(content=turn["content"]))
            elif turn.get("role") == "assistant":
                history_messages.append(AIMessage(content=turn["content"]))
        
        # Generate response
        chain = self._prompt | self._llm
        
        if stream:
            return self._stream_response(chain, context, history_messages, message, sources, retrieval_results)
        else:
            response = chain.invoke({
                "context": context,
                "history": history_messages,
                "question": message
            })
            
            confidence = 0.0
            if retrieval_results:
                confidence = sum(score for _, score in retrieval_results) / len(retrieval_results)
            
            return {
                "response": response.content,
                "sources": sources,
                "confidence": round(confidence, 3)
            }
    
    def _stream_response(
        self,
        chain,
        context: str,
        history_messages: list,
        question: str,
        sources: list,
        retrieval_results: list
    ) -> Generator[dict, None, None]:
        """Stream the response token by token."""
        full_response = ""
        
        for chunk in chain.stream({
            "context": context,
            "history": history_messages,
            "question": question
        }):
            if hasattr(chunk, 'content'):
                full_response += chunk.content
                yield {
                    "token": chunk.content,
                    "done": False
                }
        
        confidence = 0.0
        if retrieval_results:
            confidence = sum(score for _, score in retrieval_results) / len(retrieval_results)
        
        yield {
            "token": "",
            "done": True,
            "response": full_response,
            "sources": sources,
            "confidence": round(confidence, 3)
        }
    
    def get_available_models(self) -> dict:
        """Return available models."""
        return self.AVAILABLE_MODELS
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is accessible."""
        import httpx
        try:
            response = httpx.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_installed_models(self) -> list[str]:
        """Get list of models installed in Ollama."""
        import httpx
        try:
            response = httpx.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"].split(":")[0] for model in data.get("models", [])]
        except Exception:
            pass
        return []


class AudioService:
    """Service for handling audio input/output."""
    
    def __init__(self):
        """Initialize audio service."""
        self._stt_model = None
        self._tts_enabled = True
    
    def _ensure_stt_model(self):
        """Lazy load STT model."""
        if self._stt_model is None:
            from faster_whisper import WhisperModel
            self._stt_model = WhisperModel(
                "base",
                device="cpu",
                compute_type="int8",
                download_root="./models/whisper"
            )
    
    def transcribe(self, audio_path: str, language: str = "en") -> dict:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            dict with text, language, confidence
        """
        self._ensure_stt_model()
        
        segments, info = self._stt_model.transcribe(
            audio_path,
            language=language if language != "auto" else None,
            beam_size=1,
            best_of=1,
            temperature=0,
            vad_filter=True,
        )
        
        text_parts = []
        total_confidence = 0
        segment_count = 0
        
        for segment in segments:
            text_parts.append(segment.text.strip())
            total_confidence += segment.avg_logprob
            segment_count += 1
        
        full_text = " ".join(text_parts)
        
        avg_confidence = 0.0
        if segment_count > 0:
            avg_log_prob = total_confidence / segment_count
            avg_confidence = min(1.0, max(0.0, 1.0 + avg_log_prob))
        
        return {
            "text": full_text,
            "language": info.language,
            "confidence": round(avg_confidence, 3),
            "duration": round(info.duration, 2)
        }
    
    async def synthesize(self, text: str, voice: str = "en-US-AriaNeural") -> Optional[str]:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            voice: Voice identifier
            
        Returns:
            Path to generated audio file
        """
        if not self._tts_enabled:
            return None
        
        import edge_tts
        import uuid
        
        output_dir = Path("./data/tts_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_id = str(uuid.uuid4())[:8]
        output_path = output_dir / f"response_{file_id}.mp3"
        
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice)
            await communicate.save(str(output_path))
            return str(output_path)
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None


# Global service instances
chat_service = ChatService()
audio_service = AudioService()
