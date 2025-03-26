import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from textblob import TextBlob
import time

# Streamlit page configuration
st.set_page_config(page_title="Gemini Chatbot", page_icon="💬", layout="wide")

def init_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'gemini_key' not in st.session_state:
        st.session_state.gemini_key = None
    if 'openai_key' not in st.session_state:
        st.session_state.openai_key = None
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

def validate_api_keys(gemini_key, openai_key):
    """Validate Gemini and OpenAI API keys."""
    try:
        # Validate Gemini Key
        genai.configure(api_key=gemini_key)
        
        # Get available models that support generateContent
        available_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        # Validate OpenAI Key
        ChatOpenAI(api_key=openai_key, temperature=0)
        return available_models
    except Exception as e:
        st.error(f"API Key Validation Error: {e}")
        return []

def sentiment_analysis(text):
    """Perform sentiment analysis using TextBlob."""
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    if sentiment > 0:
        return "Positive", sentiment
    elif sentiment < 0:
        return "Negative", sentiment
    else:
        return "Neutral", sentiment

def get_chat_summary(conversation_history, openai_key):
    """Generate summary using OpenAI."""
    llm = ChatOpenAI(api_key=openai_key, temperature=0)
    summary_prompt = f"Summarize the following conversation in under 100 words:\n{conversation_history}"
    
    try:
        summary = llm.invoke(summary_prompt).content
        return summary
    except Exception as e:
        return f"Error generating summary: {e}"

def main():
    # Call init_session_state at the beginning of main
    init_session_state()
    # Sidebar for API Keys
    with st.sidebar:
        st.title("🔐 API Configuration")
        
        # Gemini API Key Input
        st.session_state.gemini_key = st.text_input(
            "Enter Gemini API Key", 
            type="password", 
            key="gemini_key_input"
        )
        
        # OpenAI API Key Input
        st.session_state.openai_key = st.text_input(
            "Enter OpenAI API Key", 
            type="password", 
            key="openai_key_input"
        )
        
        # Validate Button
        validate_button = st.button("Validate & Initialize Chat")

    # Main chat area
    st.title("🤖 Gemini Interactive Chatbot")

    # Chat history display
    chat_container = st.container()
    
    # User input area
    input_container = st.container()

    # Validation and Chat Initialization
    if validate_button and st.session_state.gemini_key and st.session_state.openai_key:
        # Validate API Keys
        available_models = validate_api_keys(
            st.session_state.gemini_key, 
            st.session_state.openai_key
        )
        
        if available_models:
            # Configure Gemini
            genai.configure(api_key=st.session_state.gemini_key)
            
            # Select the first available model that supports generateContent
            selected_model = next(
                (model for model in available_models if 'gemma' in model.lower()), 
                available_models[0]
            )
            
            # Initialize Langchain components
            memory = ConversationBufferMemory(return_messages=True)
            llm = ChatGoogleGenerativeAI(
                model=selected_model, 
                google_api_key=st.session_state.gemini_key
            )
            st.session_state.conversation = ConversationChain(
                llm=llm, 
                memory=memory, 
                verbose=True
            )
            
            # Clear previous chat history
            st.session_state.chat_history = []
            
            st.sidebar.success(f"Chat initialized with {selected_model}")

    # Chat Interaction
    if st.session_state.conversation:
        with input_container:
            # User input
            user_input = st.text_input(
                "Type your message...", 
                key="user_input_field"
            )
        
        # Send message
        if user_input:
            # Append user message to chat history
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )
            
            # Generate AI response
            try:
                response = st.session_state.conversation.predict(input=user_input)
                
                # Append AI response to chat history
                st.session_state.chat_history.append(
                    {"role": "ai", "content": response}
                )
            except Exception as e:
                st.error(f"Error generating response: {e}")
    
    # Display Chat History
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                # User message styling
                st.markdown(
                    f'<div style="background-color:#000; padding:10px; border-radius:10px; margin-bottom:10px;">'
                    f'<b>You:</b> {message["content"]}</div>', 
                    unsafe_allow_html=True
                )
            else:
                # AI message styling
                st.markdown(
                    f'<div style="background-color:#000; padding:10px; border-radius:10px; margin-bottom:10px;">'
                    f'<b>AI:</b> {message["content"]}</div>', 
                    unsafe_allow_html=True
                )
    
    # End Conversation Section
    if st.session_state.conversation and st.session_state.chat_history:
        if st.button("End Conversation"):
            # Combine chat history for analysis
            full_conversation = "\n".join(
                [msg['content'] for msg in st.session_state.chat_history]
            )
            
            # Get Sentiment Analysis
            sentiment, score = sentiment_analysis(full_conversation)
            st.write(f"Conversation Sentiment: {sentiment} (Score: {score})")
            
            # Generate Summary
            summary = get_chat_summary(
                full_conversation, 
                st.session_state.openai_key
            )
            st.write("Conversation Summary:")
            st.write(summary)
            
            # Reset Session
            st.session_state.chat_history = []
            st.session_state.conversation = None

if __name__ == "__main__":
    main()