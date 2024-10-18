import os
from dotenv import load_dotenv
import openai
import gradio as gr
from langchain.prompts import PromptTemplate

# Load OpenAI API key from the .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define your prompt template (you can adjust this as needed)
prompt_template = PromptTemplate(input_variables=["user_input"], template="You are a helpful assistant. {user_input}")

def call_openai(prompt):
    try:
        # Use OpenAI API (GPT-4)
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Make sure your API key supports GPT-4
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error: {str(e)}"

# Function to handle the chat
def chatbot(user_input):
    # Apply prompt engineering by concatenating the prompt template
    final_prompt = prompt_template.format(user_input=user_input)
    return call_openai(final_prompt)

# Gradio interface
with gr.Blocks() as interface:
    chatbot_ui = gr.Chatbot(label="LangChain Chatbot with OpenAI API").style(height=300)
    user_input = gr.Textbox(label="Your input", placeholder="Ask me anything...")
    submit_button = gr.Button("Submit")

    # When the user clicks submit, the chatbot function is called
    submit_button.click(fn=chatbot, inputs=user_input, outputs=chatbot_ui)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
