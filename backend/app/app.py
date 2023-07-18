from flask import Flask, request, jsonify, render_template
import openai
from data_prep import search_content
from data_prep import embed_bert_cls
import sqlite3
import csv
import numpy as np
from scipy.spatial.distance import cosine
from utils import number_sent
import torch
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

# Set up OpenAI API credentials
openai.api_key = "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"


@app.route('/')
def home():
    return render_template('index.html')

# Define endpoint for generating text
@app.route('/generate_text', methods=['POST'])
def generate_text():
    # Get prompt from request body
    data = str(request.form['area'])
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    vector = embed_bert_cls('krasava', model, tokenizer)
    nearest = search_content(number_sent, vector)
    
    nearest.append(data)
    req = ' ## '.join(nearest)
    
#     prompt = request.json['prompt']
    payload = {'prompt': str(req)}
    # Generate text using OpenAI GPT API
    def get_completion(prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message["content"] 

    # Return generated text as JSON response
    return render_template('index.html',prediction_text= get_completion(payload['prompt'])) 
#     return jsonify({'generated_text': get_completion(payload['prompt'])})

if __name__ == '__main__':
#     app.debug = True
#     app.run()
    app.run(host='0.0.0.0')
