import os
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
from fuzzywuzzy import fuzz
from google.cloud import storage
import pandas as pd
import io

model = None

# Set up Google Cloud Storage client
storage_client = storage.Client()

# Define the get_answer function
def get_answer(question, context):
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name)

    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    return answer

def handle_answer(request):
    # user_input = request.json['question']  # Retrieve JSON data from the request
    user_input = request.form.get('question')
    conversation_history = []

    if user_input:
        conversation_history.append(("User", user_input))

        # Check if the question is in the dataset
        match = None
        max_score = -1

        # Read the dataset from Cloud Storage
        bucket_name = "question-predict"
        blob_name = "data/question_answer.csv"
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        data = blob.download_as_text()

        df = pd.read_csv(io.StringIO(data))

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

    return {"answer": answer}  # Return response as dictionary

# If you want to test the function locally, you can uncomment the following lines
# and provide a sample request payload.

# sample_request = {"question": "Your question here"}
# response = handle_answer(sample_request)
# print(response)

