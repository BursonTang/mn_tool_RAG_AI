# app.py
import streamlit as st
import google.generativeai as genai
from google.cloud import storage
import pandas as pd
import numpy as np
import ast
import os
import json


# --- Configuration and API Key Setup ---
st.session_state["env_setup_done"] = False

if not st.session_state.get("env_setup_done", False):
    # --- Gemini API Key Configuratgion ---
    try:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            st.error("GEMINI_API_KEY environment variable not set. Please set it for the Gemini API.")
            st.stop()
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        st.stop()
        
    # --- Google Cloud Storage Configuration ---
    try:
        GCLOUD_JSON_STR = os.getenv("GCLOUD_JSON_STR")
        if not GCLOUD_JSON_STR:
            st.error("GCLOUD_JSON_STR environment variable not set. Please set it for the app to access Google Cloud Storage.")
            st.stop()
    except Exception as e:
        st.error(f"Error configuring Google Cloud: {e}")
        st.stop()
    
    
    # --- GCS bucket and file names configuration ---
    try: 
        GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
        GCS_FILE_NAME = os.getenv("GCS_FILE_NAME")      
        if not GCS_BUCKET_NAME or not GCS_FILE_NAME:
            st.error("GCS_BUCKET_NAME and GCS_FILE_NAME environment variables not set. Please set them for the app to access Google Cloud Storage.")
            st.stop()
    except:
            st.error(f"Error configuring Google Cloud GCS Bucket name and file name: {e}")
            st.stop()
    st.session_state["env_setup_done"] = True


# --- load and cache the database for mn_tool functions saved on Google Cloud---
# Load the data once when the app starts (or first accessed)
# This ensures it's loaded only once and cached by Streamlit for performance.
@st.cache_data # Streamlit's caching decorator
def load_data_from_gcs(bucket_name, file_name):
    """Loads data from a Google Cloud Storage bucket."""
    try:
        # Initialize a client
        credentials_info = json.loads(GCLOUD_JSON_STR)
        storage_client = storage.Client.from_service_account_info(credentials_info)
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        # For CSV (using pandas):
        # The file is downloaded to a string, then pandas reads that string.
        data_string = blob.download_as_text()
        df = pd.read_csv(pd.io.common.StringIO(data_string))
        df['Embeddings'] = df['Embeddings'].apply(ast.literal_eval)
        return df

    except Exception as e:
        st.error(f"Error loading data from GCS: {e}")
        return None # Or raise the exception
    

def find_best_passages(query, dataframe, num_results=5, embedding_model='models/text-embedding-004'):
  """
  Compute the distances between the query and each document in the dataframe
  using the dot product.
  """
  
  query_embedding = genai.embed_content(model=embedding_model,
                                        content=query,
                                        task_type="retrieval_query")
  dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
  idx = np.argsort(dot_products)[-num_results:][::-1]
  if num_results == 1:
      # Return text from index with max value
      return dataframe.iloc[idx[0]]['Text'] 
  else:
      # Return couple text with highest similarity
      return dataframe.iloc[idx]['Text'].tolist()

# --- load database df ---
df_database = load_data_from_gcs(GCS_BUCKET_NAME, GCS_FILE_NAME)
# embeddings_database = np.stack(df_database['embeddings'])

# --- Streamlit App Title ---
st.title("💬 MN Tools Inquiry Chatbot")
st.caption("🚀 Powered by Google Gemini API")

# --- Initialize Chat History in Session State ---
# Streamlit's session_state is used to persist variables across reruns
# (e.g., when a user types a new message).
if "messages" not in st.session_state:
    # Start with an initial message from the assistant
    st.session_state.messages = [{"role": "assistant", "content": "Hello! What do you want to achieve today? Maybe we have what you need in our MN Tools!"}]

# --- Display Existing Chat Messages ---
# Loop through the messages stored in session_state and display them.
for message in st.session_state.messages:
    # Use st.chat_message to display messages with appropriate avatars
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input Handling ---
# st.chat_input creates a text input field at the bottom of the chat interface.
# The `prompt` variable will contain the user's input when they press Enter.
if user_input := st.chat_input("Describe what you want to achieve with MN Tools:"):
    # Add the user's message to the chat history and display it immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Prepare Messages for Gemini API ---
    # The Gemini API expects a specific format for conversational history.
    # We convert our simple "messages" list into the required format.
    # "user" role maps to "user", "assistant" role maps to "model".
    # Each message needs a "parts" key containing a list of content.
    
    model = 'models/text-embedding-004'
    
    prompt = f"""You are a helpful and informative bot that help user find the right functions for their tasks using text from the reference passage included below. 
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
    Be sure to strike a friendly and converstional tone. If the passage is irrelevant to the answer, you may ignore it.
    Do not mention the existence of the passage in your answer, just answer the question directly.

    QUESTION: {user_input}
    """
    
    top_passages = find_best_passages(query=user_input, dataframe=df_database, num_results=5)
    

    # Add the retrieved documents to the prompt.
    for passage in top_passages:
        passage_oneline = passage.replace("\n", " ")
        prompt += f"PASSAGE: {passage_oneline}\n"

    
    messages_for_gemini = []
    for msg in st.session_state.messages:
        # Skip the current user prompt as it will be sent separately by chat.send_message
        if msg["role"] == "user" and msg["content"] == prompt:
            continue
        messages_for_gemini.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["content"]]
        })

    # --- Call Gemini API to Get Response ---
    with st.chat_message("assistant"):
        # Create a Gemini model instance. 
        model = genai.GenerativeModel('models/gemini-2.5-flash')

        # Start a chat session with the historical messages
        # The history should *not* include the current user prompt, as send_message adds it.
        chat = model.start_chat(history=messages_for_gemini)

        # Use a spinner to indicate that the AI is thinking
        with st.spinner("Thinking..."):
            try:
                # Send the current user prompt to the Gemini model
                response = chat.send_message(prompt, stream=True) # Use stream=True for streaming responses
                full_response = ""
                # Iterate over the streamed response chunks
                response_placeholder = st.empty()
                full_response = ""
                cursor = "▌"
                for chunk in response:
                    full_response += chunk.text
                    # Show the response so far with a blinking cursor
                    response_placeholder.markdown(full_response + cursor)
                # After streaming, show the final response without the cursor
                response_placeholder.markdown(full_response)
            except Exception as e:
                error_message = f"An error occurred while generating response: {e}"
                st.error(error_message)
                full_response = error_message # Store error in full_response

    # Add the assistant's response (or error message) to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

