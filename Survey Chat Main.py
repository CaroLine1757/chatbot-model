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

# Bot 2: Respondent Prompt
respondent_prompt = (
    "You are an AI respondent providing realistic and thoughtful answers to survey questions. "
    "Respond like a human participant. Limit your responses to 2 sentences. "
    "You may occasionally answer nonsense, as a human would. Diversify responses a little between conversations. "
)

# Bot 3: Evaluator Prompt
evaluator_prompt = (
    "You are an evaluator analyzing the transcript of a survey. Your job is to:\n"
    "- Determine if the survey questions stayed on topic.\n"
    "- Check if the questions included detailed follow-ups.\n"
    "- Identify if the questions were engaging and insightful.\n"
    "- Provide actionable feedback for improving the survey process.\n"
    "- Highlight any guideline violations and suggest fixes."
)

def get_topic():
    """Ask the user to specify the topic of the survey."""
    print("Bot 1: Welcome! Before we begin, could you specify the topic and objective of this survey?")
    topic = input("\nYour response (2-3 sentences about the topic and objective): ")
    return topic

def get_batch_size():
    """Ask the user to specify how many conversations to generate."""
    while True:
        try:
            batch_size = int(input("\nHow many conversations would you like to generate? "))
            if batch_size > 0:
                return batch_size
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def build_survey_prompt(topic):
    """Build the system prompt for Bot 1 based on the user-specified topic."""
    return (
        f"You are a market research analyst tasked with conducting structured, yet conversational interviews to survey participants. "
        f"Your goal is to extract deep, actionable insights that satisfy both qualitative and quantitative needs for our clients, "
        f"with a somewhat stronger emphasis on quantitative data.\n\n"
        f"Clients expect not only quantitative data but also rich qualitative insights that typically get missed in traditional surveys. "
        f"Create questions that are conversational and capable of eliciting these insights, while making participants comfortable enough "
        f"to share nuanced perspectives.\n\n"
        f"Your task:\n"
        f"- Design questions regarding participants' preferences for {topic}.\n"
        f"- Use simple, accessible language.\n"
        f"- Pose each question clearly and concisely. Limit summarizing prior responses to 5 words or fewer.\n"
        f"- You may pose up to one follow up question. \n"
        f"- Add more depth to follow-ups when ambiguity or potential key insights are noted.\n\n"
        f"- Ask one question at a time. This is a strict rule. \n\n"
        f"Brand Voice Guidelines:\n"
        f"- Tone: Educated, professional, slightly casual, friendly.\n"
        f"- Restrictions: Respond on topic only. Steer away from off-topic conversations with polite firmness."
    )

def handle_rate_limit(api_call, *args, **kwargs):
    """Handle rate limits by retrying after a delay."""
    while True:
        try:
            return api_call(*args, **kwargs)
        except openai.error.RateLimitError as e:
            print(f"Rate limit reached. Retrying in 15 seconds...")
            time.sleep(15)

def generate_survey_question(history, survey_prompt):
    """Bot 1: Generate survey questions with rate limit handling."""
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

def respond_to_question(question, history):
    """Bot 2: Respond to survey questions."""
    response = handle_rate_limit(
        openai.ChatCompletion.create,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": respondent_prompt},
            {"role": "assistant", "content": question}
        ]
    )
    answer = response['choices'][0]['message']['content']
    history.append({"role": "user", "content": answer})
    time.sleep(1)  # Add a short delay to reduce call frequency
    return answer

def evaluate_transcript(transcript):
    """Bot 3: Evaluate the conversation transcript."""
    # Format the transcript as plain text for evaluation
    transcript_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in transcript])

    # Construct the evaluation message
    evaluation_prompt = (
        f"{evaluator_prompt}\n\nTranscript:\n{transcript_text}\n\n"
        "Please provide your evaluation below. Respond in plain text."
    )

    # Get Bot 3's evaluation
    response = handle_rate_limit(
        openai.ChatCompletion.create,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": evaluator_prompt},
            {"role": "user", "content": transcript_text}
        ]
    )

    # Return the plain-text evaluation
    return response['choices'][0]['message']['content']



def run_conversation(survey_prompt):
    """Simulate one conversation between Bot 1 and Bot 2."""
    survey_history = []
    respondent_history = []
    conversation_transcript = []
    question_count = 0

    while question_count < 10:  # Limit to 10 questions
        # Bot 1 generates a question
        question = generate_survey_question(survey_history, survey_prompt)
        if "END SURVEY" in question.upper():
            print("\nBot 1 (Survey Generator): End of survey reached.")
            break

        print(f"\nBot 1 (Survey Generator): {question}")
        answer = respond_to_question(question, respondent_history)
        print(f"Bot 2 (Respondent): {answer}")

        # Append to transcript
        conversation_transcript.append({"role": "Survey Generator", "content": question})
        conversation_transcript.append({"role": "Respondent", "content": answer})
        question_count += 1

    return conversation_transcript

def evaluate_all_transcripts(transcripts):
    """
    Bot 3: Evaluate the entire batch of conversations at once.
    Provide the concatenated conversation history to Bot 3 and ask for a comprehensive evaluation.
    """
    full_transcript = ""
    for i, conversation in enumerate(transcripts):
        full_transcript += f"\n--- Conversation {i + 1} ---\n"
        full_transcript += "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        full_transcript += "\n"

    evaluation_prompt = (
        f"{evaluator_prompt}\n\nHere is the full transcript of all conversations:\n{full_transcript}\n\n"
        "Please provide a detailed and consolidated evaluation of the entire batch of conversations. "
        "Focus on overall trends, strengths, weaknesses, and actionable feedback."
    )

    response = handle_rate_limit(
        openai.ChatCompletion.create,
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": evaluator_prompt},
            {"role": "user", "content": full_transcript}
        ]
    )

    return response['choices'][0]['message']['content']

def run_batch_conversations(batch_size, survey_prompt):
    """Run multiple conversations and conduct a batch evaluation."""
    all_transcripts = []

    for i in range(batch_size):
        print(f"\n--- Running Conversation {i + 1} ---")
        transcript = run_conversation(survey_prompt)
        all_transcripts.append(transcript)

    consolidated_evaluation = evaluate_all_transcripts(all_transcripts)

    print("\n--- Consolidated Evaluation ---")
    print(consolidated_evaluation)

    return all_transcripts, consolidated_evaluation

def main():
    print("Welcome to the Survey Simulation!")
    topic = get_topic()
    batch_size = get_batch_size()
    survey_prompt = build_survey_prompt(topic)
    transcripts, evaluation = run_batch_conversations(batch_size, survey_prompt)

if __name__ == "__main__":
    main()