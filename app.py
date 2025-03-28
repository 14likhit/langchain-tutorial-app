import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from textblob import TextBlob
import requests
import time
from io import BytesIO
from PIL import Image

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
    if 'huggingface_key' not in st.session_state:
        st.session_state.huggingface_key = None
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'generated_image' not in st.session_state:
        st.session_state.generated_image = None

def validate_api_keys(gemini_key, openai_key, huggingface_key=None):
    """Validate Gemini, OpenAI, and Hugging Face API keys."""
    try:
        # Validate Gemini Key
        genai.configure(api_key=gemini_key)
        available_models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        
        # Validate OpenAI Key
        ChatOpenAI(api_key=openai_key, temperature=0)
        
        # Validate Hugging Face Key (optional, only if provided)
        if huggingface_key:
            test_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
            headers = {"Authorization": f"Bearer {huggingface_key}"}
            response = requests.get(test_url, headers=headers)
            if response.status_code != 200:
                raise Exception("Invalid Hugging Face API key")
        
        return available_models
    except Exception as e:
        st.error(f"API Key Validation Error: {e}")
        return []

def sentiment_analysis(text):
    """Perform sentiment analysis using TextBlob."""
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral", sentiment

def get_chat_summary(conversation_history, openai_key):
    """Generate summary using OpenAI."""
    llm = ChatOpenAI(api_key=openai_key, temperature=0)
    summary_prompt = f"Summarize the following conversation in under 100 words:\n{conversation_history}"
    try:
        summary = llm.invoke(summary_prompt).content
        return summary
    except Exception as e:
        return f"Error generating summary: {e}"

def generate_image_from_summary(summary, huggingface_key):
    """Generate an image from the summary using Hugging Face API."""
    model_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
    headers = {"Authorization": f"Bearer {huggingface_key}"}
    payload = {"inputs": summary}
    
    try:
        response = requests.post(model_url, headers=headers, json=payload)
        if response.status_code == 200:
            image_bytes = response.content
            image = Image.open(BytesIO(image_bytes))
            return image
        else:
            st.error(f"Image generation failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

def main():
    init_session_state()
    
    # Sidebar for API Keys
    with st.sidebar:
        st.title("🔐 API Configuration")
        
        st.session_state.gemini_key = st.text_input("Enter Gemini API Key", type="password", key="gemini_key_input")
        st.session_state.openai_key = st.text_input("Enter OpenAI API Key", type="password", key="openai_key_input")
        st.session_state.huggingface_key = st.text_input("Enter Hugging Face API Key", type="password", key="huggingface_key_input")
        
        validate_button = st.button("Validate & Initialize Chat")

    # Main chat area
    st.title("🤖 Gemini Interactive Chatbot")

    chat_container = st.container()
    input_container = st.container()

    # Validation and Chat Initialization
    if validate_button and st.session_state.gemini_key and st.session_state.openai_key:
        available_models = validate_api_keys(
            st.session_state.gemini_key, 
            st.session_state.openai_key,
            st.session_state.huggingface_key
        )
        
        if available_models:
            genai.configure(api_key=st.session_state.gemini_key)
            selected_model = next(
                (model for model in available_models if 'gemma' in model.lower()), 
                available_models[0]
            )
            memory = ConversationBufferMemory(return_messages=True)
            llm = ChatGoogleGenerativeAI(model=selected_model, google_api_key=st.session_state.gemini_key)
            st.session_state.conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
            st.session_state.chat_history = []
            st.sidebar.success(f"Chat initialized with {selected_model}")

    # Chat Interaction
    if st.session_state.conversation:
        with input_container:
            user_input = st.text_input("Type your message...", key="user_input_field")
        
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            try:
                response = st.session_state.conversation.predict(input=user_input)
                st.session_state.chat_history.append({"role": "ai", "content": response})
            except Exception as e:
                st.error(f"Error generating response: {e}")
    
    # Display Chat History
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(
                    f'<div style="background-color:#000; padding:10px; border-radius:10px; margin-bottom:10px;">'
                    f'<b>You:</b> {message["content"]}</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="background-color:#000; padding:10px; border-radius:10px; margin-bottom:10px;">'
                    f'<b>AI:</b> {message["content"]}</div>', 
                    unsafe_allow_html=True
                )
    
    # End Conversation Section
    if st.session_state.conversation and st.session_state.chat_history:
        if st.button("End Conversation"):
            full_conversation = "\n".join([msg['content'] for msg in st.session_state.chat_history])
            sentiment, score = sentiment_analysis(full_conversation)
            st.write(f"Conversation Sentiment: {sentiment} (Score: {score})")
            
            summary = get_chat_summary(full_conversation, st.session_state.openai_key)
            st.write("Conversation Summary:")
            st.write(summary)
            
            # Generate and display image if Hugging Face key is provided
            if st.session_state.huggingface_key:
                st.write("Generating image from summary...")
                image = generate_image_from_summary(summary, st.session_state.huggingface_key)
                if image:
                    st.session_state.generated_image = image
                    st.image(image, caption="Generated Image from Summary", use_column_width=True)
            
            st.session_state.chat_history = []
            st.session_state.conversation = None

if __name__ == "__main__":
    main()