import pandas as pd
from flask import Flask, render_template, request
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from fuzzywuzzy import fuzz
import torch

app = Flask(__name__)

# Read the dataset
df = pd.read_csv('../question_answer.csv')

# Set up the model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Define the get_answer function
def get_answer(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

@app.route('/')
def home():
    return "API Service"

@app.route('/answer', methods=['POST'])
def get_question_answer():
    user_input = request.form['question']
    conversation_history = []

    if user_input:
        conversation_history.append(("User", user_input))

        # Check if the question is in the dataset
        match = None
        max_score = -1
        for i, question in enumerate(df['Question']):
            score = fuzz.token_sort_ratio(user_input, question)
            if score > max_score:
                max_score = score
                match = df.iloc[i]

        # If the score is high enough, display the answer
        if max_score > 70:
            context = match['Answer']
            answer = get_answer(user_input, context)
        # If the score is not high enough, call the agent
        else:
            answer = "I'm sorry, my data is not trained for that question. I will call the agent"

        conversation_history.append(("Chatbot", answer))

    return { "answer" : answer }, 200

if __name__ == '__main__':
    # app.run()
    app.run(host="localhost", port=9000)