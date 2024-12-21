import openai
from dotenv import load_dotenv
import os
import time
from collections import Counter

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Verify API key is loaded
if not openai.api_key:
    raise ValueError("Failed to load OpenAI API key. Make sure your .env file is named '.env' and properly formatted.")

# Initializes conversation history
conversation_history = []
survey_responses = []

# System messages
survey_system_message = (
    "You are an AI-powered market research assistant designed to interact with participants "
    "in structured, yet conversational surveys, aiming to extract deep, actionable insights. "
    "Respond professionally, conversationally, and keep responses strictly relevant to the survey topic. "
)
boss_system_message = (
    "You are an AI assistant analyst summarizing key insights from multiple survey responses. "
    "Provide analysis with major themes, sentiment, and actionable recommendations, "
    "backed by data wherever possible."
)

# Add initial system message to conversation
conversation_history.append({"role": "system", "content": survey_system_message})

# Set default role
current_role = "Surveyed User"


def get_response(user_message, role):
    try:
        if role == "Surveyed User":
            # Add user's message to conversation history
            conversation_history.append({"role": "user", "content": user_message})
            survey_responses.append(user_message)

            # Get ChatGPT response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation_history
            )
            assistant_message = response["choices"][0]["message"]["content"]
            conversation_history.append({"role": "assistant", "content": assistant_message})

        elif role == "Boss":
            # Create analysis prompt
            analysis_prompt = boss_system_message + "\n\nCollected responses:\n"
            for i, response in enumerate(survey_responses, start=1):
                analysis_prompt += f"{i}. {response}\n"
            analysis_prompt += "\nProvide analysis based on the responses above."

            # Get analysis response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            assistant_message = response["choices"][0]["message"]["content"]

        return assistant_message

    except openai.error.OpenAIError as e:
        return f"An error occurred with the OpenAI API: {str(e)}"


# Main interactive loop
print("Starting the interactive ChatGPT terminal. Type 'switch' to change roles or 'exit' to quit.")

while True:
    try:
        if current_role == "Surveyed User":
            user_message = input("\nSurveyed User Response (or type 'switch' to change role): ")
            if user_message.lower() == "exit":
                print("Exiting the chat. Goodbye!")
                break
            elif user_message.lower() == "switch":
                current_role = "Boss"
                print("\nRole switched to Boss.")
            else:
                response = get_response(user_message, current_role)
                print(f"\nChatGPT (Surveyor): {response}")

        elif current_role == "Boss":
            user_message = input("\nBoss Request (e.g., 'summarize insights so far') (or type 'switch' to change role): ")
            if user_message.lower() == "exit":
                print("Exiting the chat. Goodbye!")
                break
            elif user_message.lower() == "switch":
                current_role = "Surveyed User"
                print("\nRole switched to Surveyed User.")
            else:
                response = get_response(user_message, current_role)
                print(f"\nChatGPT (Analysis): {response}")
        else:
            print("Invalid role. Resetting to 'Surveyed User'.")
            current_role = "Surveyed User"

    except KeyboardInterrupt:
        print("\nSession terminated by user. Goodbye!")
        break
