"""
RAG (Retrieval-Augmented Generation) Service.

Implements the core knowledge retrieval and response generation pipeline:
1. Document ingestion and chunking
2. Vector embedding and storage
3. Semantic retrieval
4. LLM-based response generation with context

Architecture optimized for accuracy (no hallucination) and low latency.
Supports local deployment with Ollama (FREE) or cloud providers.
"""

import logging
from pathlib import Path
from typing import Optional
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain_core.messages import HumanMessage, AIMessage

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# System prompt designed to prevent hallucination and ensure accuracy
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


class RAGService:
    """
    RAG Service for intelligent knowledge retrieval and response generation.
    
    Architecture Decisions:
    
    1. Vector Store (ChromaDB):
       - Embedded/local database for prototype simplicity
       - Persistent storage for quick restarts
       - Easy to migrate to Pinecone/Weaviate for production
    
    2. Embeddings (Sentence Transformers - all-MiniLM-L6-v2):
       - Open source, runs locally (no API costs)
       - 384 dimensions - good accuracy/speed balance
       - Optimized for semantic search
    
    3. Text Splitting:
       - RecursiveCharacterTextSplitter preserves document structure
       - 500 char chunks with 50 overlap for context continuity
       - Optimized for government service documents
    
    4. LLM (Configurable - Default: Ollama):
       - Ollama (default): FREE, local, runs on M1 Mac
       - Groq: Free tier, fast cloud inference
       - OpenAI: Production-ready, high quality
    
    5. Retrieval Strategy:
       - Similarity search with k=4 for balanced context
       - Relevance scoring for confidence calculation
    """
    
    _instance: Optional["RAGService"] = None
    
    def __new__(cls):
        """Singleton to avoid reloading models."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize RAG components."""
        if self._initialized:
            return
            
        self._initialize_embeddings()
        self._initialize_vector_store()
        self._initialize_llm()
        self._initialized = True
    
    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        logger.info(f"Loading embedding model: {settings.embedding_model} (this may take a moment on first run)...")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": settings.stt_device},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _initialize_vector_store(self):
        """Initialize or load the vector store."""
        persist_dir = Path(settings.chroma_persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(persist_dir)
        )
        
        # Check if we need to ingest knowledge base
        collection = self.vector_store._collection
        if collection.count() == 0:
            logger.info("Empty vector store - ingesting knowledge base")
            self._ingest_knowledge_base()
        else:
            logger.info(f"Loaded vector store with {collection.count()} documents")
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration."""
        logger.info(f"Initializing LLM: {settings.llm_provider}/{settings.llm_model}")
        
        if settings.llm_provider == "ollama":
            # Determine the correct base URL
            # Use host.docker.internal when running in Docker, localhost otherwise
            base_url = os.environ.get(
                "OLLAMA_BASE_URL", 
                settings.ollama_base_url_local  # Default to localhost for local dev
            )
            logger.info(f"Connecting to Ollama at: {base_url}")
            
            self.llm = ChatOllama(
                model=settings.llm_model,
                base_url=base_url,
                temperature=settings.llm_temperature,
                num_predict=settings.llm_max_tokens,
            )
        elif settings.llm_provider == "groq":
            from langchain_groq import ChatGroq
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY environment variable required")
            self.llm = ChatGroq(
                api_key=settings.groq_api_key,
                model_name=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens
            )
        elif settings.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable required")
            self.llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model_name=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
        
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        logger.info("LLM initialized successfully")
    
    def _ingest_knowledge_base(self):
        """Ingest all documents from the knowledge directory."""
        knowledge_dir = Path(settings.knowledge_directory)
        
        if not knowledge_dir.exists():
            logger.warning(f"Knowledge directory not found: {knowledge_dir}")
            return
        
        # Load all text files
        loader = DirectoryLoader(
            str(knowledge_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        logger.info("Documents ingested into vector store")
    
    def ingest_document(self, content: str, filename: str) -> int:
        """
        Ingest a single document into the vector store.
        
        Args:
            content: Document content
            filename: Document identifier
            
        Returns:
            Number of chunks created
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Create document with metadata
        doc = Document(page_content=content, metadata={"source": filename})
        chunks = text_splitter.split_documents([doc])
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        logger.info(f"Ingested {filename}: {len(chunks)} chunks")
        return len(chunks)
    
    async def query(
        self,
        question: str,
        conversation_history: list[dict] = None
    ) -> dict:
        """
        Process a query and generate a response.
        
        Args:
            question: User's question
            conversation_history: Previous conversation turns for context
            
        Returns:
            dict with:
            - response: Generated answer
            - sources: List of source documents with relevance scores
            - confidence: Overall confidence score
            
        Latency Optimization:
        - Retrieval uses approximate nearest neighbor search
        - LLM streaming available for progressive response
        - Conversation history limited to prevent context bloat
        """
        conversation_history = conversation_history or []
        
        # Retrieve relevant documents
        retrieval_results = self.vector_store.similarity_search_with_relevance_scores(
            question,
            k=settings.retrieval_k
        )
        
        # Format context from retrieved documents
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
        
        # Format conversation history for the prompt
        history_messages = []
        for turn in conversation_history[-settings.max_history_length:]:
            if turn["role"] == "user":
                history_messages.append(HumanMessage(content=turn["content"]))
            else:
                history_messages.append(AIMessage(content=turn["content"]))
        
        # Generate response
        chain = self.prompt | self.llm
        
        response = await chain.ainvoke({
            "context": context,
            "history": history_messages,
            "question": question
        })
        
        # Calculate confidence based on source relevance
        confidence = 0.0
        if retrieval_results:
            confidence = sum(score for _, score in retrieval_results) / len(retrieval_results)
        
        return {
            "response": response.content,
            "sources": sources,
            "confidence": round(confidence, 3)
        }
    
    def clear_vector_store(self):
        """Clear all documents from the vector store."""
        self.vector_store._collection.delete(where={})
        logger.info("Vector store cleared")
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        return self.vector_store._collection.count()
    
    def is_healthy(self) -> bool:
        """Check if RAG service is operational."""
        return (
            self._initialized and 
            self.embeddings is not None and 
            self.vector_store is not None and 
            self.llm is not None
        )


# Global service instance
rag_service = RAGService()
