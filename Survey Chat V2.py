import openai
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Verify API key is loaded
if not openai.api_key:
    raise ValueError("Failed to load OpenAI API key. Make sure your .env file is named '.env' and properly formatted.")

# Define the five questions
survey_questions = [
    "Who is your target audience for this survey?",
    "How much time do you think respondents should spend on the survey?",
    "What topic or domain do you want the survey to focus on?",
    "What kind of insights or outcomes do you hope to achieve from the survey?",
    "What specific questions would you like me to ask?"
]

def get_survey_inputs():
    """Ask the user the five predefined questions."""
    responses = {}
    for idx, question in enumerate(survey_questions, start=1):
        print(f"Bot 1: {question}")
        response = input(f"Your response to Question {idx}: ")
        responses[f"Q{idx}"] = response
    return responses

def build_survey_prompt(survey_inputs):
    """Build the system prompt for Bot 1 based on the user-provided inputs."""
    return (
        "You are a market research analyst tasked with conducting structured, yet conversational interviews to survey participants. "
        "Your goal is to extract deep, actionable insights that satisfy both qualitative and quantitative needs for our clients.\n\n"
        "Clients expect not only quantitative data but also rich qualitative insights that typically get missed in traditional surveys. "
        "Create questions that are conversational and capable of eliciting these insights, while making participants comfortable enough "
        "to share nuanced perspectives.\n\n"
        "Your task:\n"
        f"- Design questions regarding participants' preferences for {survey_inputs['Q3']}.\n"
        f"- Target audience: {survey_inputs['Q1']}.\n"
        f"- Desired insights: {survey_inputs['Q4']}.\n"
        f"- Expected respondent time: {survey_inputs['Q2']} minutes.\n"
        f"- Consider these specific questions: {survey_inputs['Q5']}.\n"
        "Guidelines:\n"
        "- Use simple, accessible language.\n"
        "- Pose each question clearly and concisely. Limit summarizing prior responses to 5 words or fewer.\n"
        "- Pay attention to this rule. Ask one question at a time. This is a strict rule.\n"
        "- Add depth to follow-ups when ambiguity or potential key insights are noted.\n\n"
        "- Ensure that your output contains exactly one question per response. If it includes multiple questions, reformat and provide only one question. \n"
        "Tone:\n"
        "- Educated, professional, slightly casual, friendly.\n"
        "- Respond on topic only. Steer away from off-topic conversations with polite firmness."
    )

def handle_rate_limit(api_call, *args, **kwargs):
    """Handle rate limits by retrying after a delay."""
    while True:
        try:
            return api_call(*args, **kwargs)
        except openai.error.RateLimitError as e:
            print("Rate limit reached. Retrying in 15 seconds...")
            time.sleep(15)

def generate_survey_question(history, survey_prompt):
    """Bot 1: Generate one survey question with rate limit handling."""
    response = handle_rate_limit(
        openai.ChatCompletion.create,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": survey_prompt},
            *history
        ]
    )
    question = response['choices'][0]['message']['content']
    history.append({"role": "assistant", "content": question})
    time.sleep(1)  # Add a short delay to reduce call frequency
    return question

def insert_specific_questions(history, user_defined_questions):
    """Decide when to insert user-specific questions into the conversation."""
    if user_defined_questions and len(history) % 3 == 0:  # Insert every 3rd question
        question = user_defined_questions.pop(0)
        history.append({"role": "assistant", "content": question})
        return question
    return None

def run_conversation(survey_prompt, user_defined_questions):
    """Simulate the survey conversation after setup questions."""
    survey_history = []

    while True:
        question = generate_survey_question(survey_history, survey_prompt)
        if "END SURVEY" in question.upper():
            print("\nBot 1 (Survey Generator): End of survey reached.")
            break

        print(f"\nBot 1 (Survey Generator): {question}")
        user_response = input("Your response: ")
        survey_history.append({"role": "user", "content": user_response})

def main():
    print("Welcome to the Survey Simulation!")
    # Ask the setup questions first
    survey_inputs = get_survey_inputs()
    user_defined_questions = survey_inputs['Q5'].split(';') if survey_inputs['Q5'] else []
    survey_prompt = build_survey_prompt(survey_inputs)

    # Start the survey
    run_conversation(survey_prompt, user_defined_questions)

if __name__ == "__main__":
    main()
