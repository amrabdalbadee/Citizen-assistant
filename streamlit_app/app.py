"""
🏛️ Citizen Support Assistant - Streamlit UI

A beautiful, multimodal chat interface for government services.
Features:
- Text and voice input
- Audio responses (TTS)
- Conversation history with database storage
- Model selection
- Source citations
"""

import streamlit as st
import asyncio
from pathlib import Path
import sys
import time
from datetime import datetime
import base64

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from database.models import ConversationDB
from services.chat_service import ChatService, AudioService

# Page configuration
st.set_page_config(
    page_title="Citizen Support Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        max-width: 80%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Sources expander */
    .sources-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Conversation list */
    .conversation-item {
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: background 0.2s;
    }
    
    .conversation-item:hover {
        background: #e9ecef;
    }
    
    .conversation-item.active {
        background: #667eea;
        color: white;
    }
    
    /* Audio player */
    audio {
        width: 100%;
        margin-top: 0.5rem;
    }
    
    /* Model selector */
    .model-badge {
        background: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    
    /* Header */
    .app-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    
    /* Confidence indicator */
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    
    /* Input area */
    .stTextArea textarea {
        border-radius: 15px;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 20px;
        padding: 0.5rem 2rem;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_model" not in st.session_state:
        st.session_state.current_model = "llama3.2"
    if "tts_enabled" not in st.session_state:
        st.session_state.tts_enabled = True
    if "db" not in st.session_state:
        st.session_state.db = ConversationDB()
    if "chat_service" not in st.session_state:
        st.session_state.chat_service = ChatService()
    if "audio_service" not in st.session_state:
        st.session_state.audio_service = AudioService()
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True


def get_confidence_class(confidence: float) -> str:
    """Get CSS class for confidence level."""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    return "confidence-low"


def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%b %d, %H:%M")
    except:
        return timestamp


def render_sidebar():
    """Render the sidebar with conversation history and settings."""
    with st.sidebar:
        st.markdown("## 🏛️ Citizen Support")
        st.markdown("---")
        
        # New conversation button
        if st.button("➕ New Conversation", use_container_width=True, type="primary"):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        
        # Model selection
        st.markdown("### 🤖 AI Model")
        
        available_models = st.session_state.chat_service.get_available_models()
        installed_models = st.session_state.chat_service.get_installed_models()
        
        model_options = []
        model_labels = {}
        
        for model_id, model_info in available_models.items():
            is_installed = any(model_id.split(":")[0] in m for m in installed_models)
            status = "✅" if is_installed else "⬇️"
            label = f"{status} {model_info['name']} ({model_info['size']})"
            model_options.append(model_id)
            model_labels[model_id] = label
        
        selected_model = st.selectbox(
            "Select Model",
            options=model_options,
            format_func=lambda x: model_labels.get(x, x),
            index=model_options.index(st.session_state.current_model) if st.session_state.current_model in model_options else 0,
            help="Models with ✅ are installed. Models with ⬇️ need to be pulled first."
        )
        
        if selected_model != st.session_state.current_model:
            st.session_state.current_model = selected_model
            st.session_state.chat_service.switch_model(selected_model)
            st.toast(f"Switched to {available_models[selected_model]['name']}")
        
        # Show model info
        if selected_model in available_models:
            st.caption(available_models[selected_model]['description'])
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        
        st.session_state.tts_enabled = st.toggle(
            "🔊 Voice Responses",
            value=st.session_state.tts_enabled,
            help="Enable text-to-speech for responses"
        )
        
        st.session_state.show_sources = st.toggle(
            "📚 Show Sources",
            value=st.session_state.show_sources,
            help="Show source documents used for answers"
        )
        
        st.markdown("---")
        
        # Conversation history
        st.markdown("### 💬 Conversations")
        
        # Search
        search_query = st.text_input("🔍 Search", placeholder="Search conversations...")
        
        # List conversations
        if search_query:
            conversations = st.session_state.db.search_conversations(search_query)
        else:
            conversations = st.session_state.db.list_conversations(limit=20)
        
        for conv in conversations:
            is_active = conv["id"] == st.session_state.conversation_id
            
            col1, col2 = st.columns([5, 1])
            
            with col1:
                title = conv["title"][:30] + "..." if len(conv["title"]) > 30 else conv["title"]
                btn_type = "primary" if is_active else "secondary"
                
                if st.button(
                    f"{'🔹' if is_active else '💬'} {title}",
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                    type=btn_type
                ):
                    load_conversation(conv["id"])
            
            with col2:
                if st.button("🗑️", key=f"del_{conv['id']}", help="Delete"):
                    st.session_state.db.delete_conversation(conv["id"])
                    if st.session_state.conversation_id == conv["id"]:
                        st.session_state.conversation_id = None
                        st.session_state.messages = []
                    st.rerun()
        
        if not conversations:
            st.caption("No conversations yet. Start chatting!")
        
        st.markdown("---")
        
        # Ollama status
        ollama_connected = st.session_state.chat_service.check_ollama_connection()
        if ollama_connected:
            st.success("🟢 Ollama Connected")
        else:
            st.error("🔴 Ollama Disconnected")
            st.caption("Make sure Ollama is running: `ollama serve`")


def load_conversation(conversation_id: str):
    """Load a conversation from the database."""
    conversation = st.session_state.db.get_conversation_with_messages(conversation_id)
    if conversation:
        st.session_state.conversation_id = conversation_id
        st.session_state.messages = conversation.get("messages", [])
        
        # Set model if saved
        if conversation.get("model_used"):
            st.session_state.current_model = conversation["model_used"]
            st.session_state.chat_service.switch_model(conversation["model_used"])
    
    st.rerun()


def render_message(message: dict, index: int):
    """Render a single message."""
    role = message.get("role", "user")
    content = message.get("content", "")
    audio_path = message.get("audio_path")
    sources = message.get("sources", [])
    confidence = message.get("confidence")
    
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>You</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        confidence_class = get_confidence_class(confidence) if confidence else ""
        confidence_html = f'<span class="{confidence_class}">Confidence: {confidence:.0%}</span>' if confidence else ""
        
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🏛️ Assistant</strong> {confidence_html}<br>
            {content}
        </div>
        """, unsafe_allow_html=True)
        
        # Audio player
        if audio_path and Path(audio_path).exists():
            st.audio(audio_path, format="audio/mp3")
        
        # Sources
        if sources and st.session_state.show_sources:
            with st.expander("📚 Sources", expanded=False):
                for i, source in enumerate(sources):
                    st.markdown(f"""
                    **{source.get('source', 'Unknown')}** (Relevance: {source.get('relevance_score', 0):.0%})
                    
                    > {source.get('content', '')}
                    """)
                    if i < len(sources) - 1:
                        st.markdown("---")


def render_chat():
    """Render the main chat interface."""
    # Header
    st.markdown("""
    <div class="app-header">
        <h1>🏛️ Citizen Support Assistant</h1>
        <p>Ask questions about government services. I'll help you find accurate information.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.messages:
            # Welcome message
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #666;">
                <h3>👋 Welcome!</h3>
                <p>I can help you with:</p>
                <ul style="list-style: none; padding: 0;">
                    <li>📄 Passport applications and renewals</li>
                    <li>📋 Birth certificate requests</li>
                    <li>💰 Fees and processing times</li>
                    <li>📍 Office locations and hours</li>
                </ul>
                <p><em>Type your question below or use voice input 🎤</em></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Render messages
            for i, message in enumerate(st.session_state.messages):
                render_message(message, i)
    
    # Input area
    st.markdown("---")
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_area(
            "Your message",
            placeholder="Ask about passport applications, birth certificates, fees, requirements...",
            height=80,
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        send_button = st.button("Send 📤", type="primary", use_container_width=True)
    
    # Voice input
    col3, col4, col5 = st.columns([2, 2, 2])
    
    with col3:
        audio_file = st.file_uploader(
            "🎤 Upload audio",
            type=["wav", "mp3", "ogg", "m4a"],
            key="audio_upload",
            label_visibility="collapsed",
            help="Upload a voice recording"
        )
    
    with col4:
        if audio_file:
            if st.button("Transcribe & Send 🎙️", use_container_width=True):
                process_audio_input(audio_file)
    
    with col5:
        voice_options = ["en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural", "en-GB-RyanNeural"]
        selected_voice = st.selectbox(
            "Voice",
            voice_options,
            index=0,
            label_visibility="collapsed",
            help="Select TTS voice"
        )
        st.session_state.tts_voice = selected_voice
    
    # Process text input
    if send_button and user_input.strip():
        process_text_input(user_input.strip())


def process_text_input(user_input: str):
    """Process text input and generate response."""
    # Create conversation if needed
    if not st.session_state.conversation_id:
        st.session_state.conversation_id = st.session_state.db.create_conversation(
            title=user_input[:50],
            model_used=st.session_state.current_model
        )
    
    # Add user message
    st.session_state.db.add_message(
        st.session_state.conversation_id,
        "user",
        user_input
    )
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Generate response
    with st.spinner("Thinking..."):
        try:
            # Get conversation history for context
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]  # Exclude current message
            ]
            
            result = st.session_state.chat_service.chat(
                user_input,
                conversation_history=history
            )
            
            response = result["response"]
            sources = result.get("sources", [])
            confidence = result.get("confidence", 0)
            
            # Generate audio if enabled
            audio_path = None
            if st.session_state.tts_enabled:
                voice = getattr(st.session_state, 'tts_voice', 'en-US-AriaNeural')
                audio_path = asyncio.run(
                    st.session_state.audio_service.synthesize(response, voice)
                )
            
            # Save assistant message
            st.session_state.db.add_message(
                st.session_state.conversation_id,
                "assistant",
                response,
                audio_path=audio_path,
                sources=sources,
                confidence=confidence
            )
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "audio_path": audio_path,
                "sources": sources,
                "confidence": confidence
            })
            
            # Auto-title conversation
            if len(st.session_state.messages) == 2:  # First exchange
                st.session_state.db.auto_title_conversation(
                    st.session_state.conversation_id,
                    user_input
                )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I apologize, but I encountered an error: {str(e)}. Please make sure Ollama is running.",
                "sources": [],
                "confidence": 0
            })
    
    st.rerun()


def process_audio_input(audio_file):
    """Process audio input, transcribe, and generate response."""
    with st.spinner("Transcribing audio..."):
        try:
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            
            # Transcribe
            result = st.session_state.audio_service.transcribe(tmp_path)
            transcribed_text = result.get("text", "").strip()
            
            # Clean up
            Path(tmp_path).unlink()
            
            if transcribed_text:
                st.success(f"Transcribed: {transcribed_text}")
                process_text_input(transcribed_text)
            else:
                st.error("Could not transcribe audio. Please try again with clearer speech.")
                
        except Exception as e:
            st.error(f"Transcription error: {str(e)}")


def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
