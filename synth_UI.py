import streamlit as st
import anthropic
from db import MongoEngine

# Assuming claude library is used to interface with Claude API and initialized similarly to OpenAI's library
client = anthropic.Anthropic(api_key=st.secrets["api_key"])
mongo_engine = MongoEngine()

instructions = """You are an AI assistant helping users order custom synthetic datasets. 

Focus only on collecting information from users for synthetic data generation. If you receive a prompt unrelated to synthetic data generation, reply with answers like "I am sorry, but I can only help you with synthetic data generation."

Please initiate a friendly conversation flow to gather the necessary details
        
Key points in your information collection process:
        
        - Ask for purpose, what is the problem you want to resolve with AI ****** function a
        - Classify if purpose is classification or a fine-tuning task. If problem is not well-defined ask for more. (non textual data, generalized tasks)
        - if its classification, suggest a set of labels, and ask for custom labels if needed
        - if its fine-tuning, suggest the fine-tuning task type, and ask for custom task type if needed or allow user to force select another type 
        - Ask for alignment, alignment for particular human preference/values: create alignment examples based on the task
        - ask for language
        - ask dataset size
        - ask for confirmation by summarizing the request

Keep questions firm and concise, as questions in multiple turns if necessary. Ask only a single question in each term. Maximize the efficiency in questions. Output final dataset request as a JSON file with fields [purpose, task_type, language, alignment_preferences:list, dataset_size]. Only output the JSON, nothing else."""


# Getting Response from Anthropic's LLM
def get_llm_response(chat_context, user_examples, user_query):
    """Get a response from the LLM using the chat history and knowledge list."""
    system_prompt = instructions
    user_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Chat history: {chat_context}\nUser query: {user_query}\nExamples: {user_examples}"
                }
            ]
        }
    ]

    try:
        # Send the message to the Anthropic API and capture the response
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0,
            messages=user_messages,
            system=system_prompt
        )

        # Extract text from the response if available
        if message and hasattr(message, 'content'):
            response_texts = [getattr(block, 'text', 'No text found') for block in message.content]
            return ' '.join(response_texts)
        elif hasattr(message, 'error'):
            error_message = getattr(message.error, 'message', 'Unknown error')
            return f"Error: {error_message}"
        else:
            return "No response from Claude."

    except Exception as e:
        # Log the exception for debugging
        print(f"Failed to get a response: {str(e)}")
        return "Failed to process your request."



# Streamlit App
st.title("Request Synthetic Data (A)")

col1, col2, col3 = st.columns([6, 1, 2])

# Initialize chat messages and user entries in session state
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'user_examples' not in st.session_state:
    st.session_state.user_examples = ["", "", ""]
if 'submit_flag' not in st.session_state:
    st.session_state.submit_flag = False

with col1:
    for message in st.session_state.chat_messages:
        st.write(message)

    user_query = st.text_input("Enter your query:", key="user_query")
    submit_button = st.button("Submit", on_click=lambda: setattr(st.session_state, 'submit_flag', True))

with col3:
    st.write("You can provide up to 3 examples:")
    for i in range(3):
        st.session_state.user_examples[i] = st.text_input(f"Example {i+1}:", key=f"example_{i}", value=st.session_state.user_examples[i])

if submit_button and user_query and st.session_state.submit_flag:
    # Append user query to chat context
    st.session_state.chat_messages.append(f"User: {user_query}")
    # Get response from LLM
    response = get_llm_response(st.session_state.chat_messages[-10:], st.session_state.user_examples, user_query)
    mongo_engine.save_form(response)
    st.session_state.chat_messages.append(f"AI: {response}")
    # Reset flag to prevent duplicate messages
    st.session_state.submit_flag = False
    # Refresh the page to show new messages
    st.experimental_rerun()
