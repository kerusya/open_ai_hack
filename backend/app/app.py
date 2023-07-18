from flask import Flask, request, jsonify, render_template
import openai
from data_prep import search_content
from data_prep import embed_bert_cls
import sqlite3
import csv
import numpy as np
from scipy.spatial.distance import cosine
from utils import number_sent
from utils import changed_schema_description
from utils import new_text_schema_description
from utils import template_string
from utils import explanation_schema_description
import torch
from transformers import AutoTokenizer, AutoModel
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json


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
    vector = embed_bert_cls(data, model, tokenizer)
    nearest, ids = search_content(number_sent, vector)
    
    chat = ChatOpenAI(openai_api_key=openai.api_key, temperature=0)
    changed_schema = ResponseSchema(name='Changed', description=changed_schema_description)
    new_text_schema = ResponseSchema(name='New_text', description=new_text_schema_description)
    explanation_schema = ResponseSchema(name='Explanation', description=explanation_schema_description)
    response_schemas = [changed_schema, new_text_schema, explanation_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    return_dict = {}
    a = 0
    for i, document in enumerate(nearest):
        a +=1
        prompt_template = ChatPromptTemplate.from_template(template_string)
        message = prompt_template.format_messages(original_document=document,
                                              new_document=data,
                                              format_instructions=format_instructions)


        response = chat(message)
        output_dict = output_parser.parse(response.content)
        return_dict[ids[i]] = output_dict
        
    initial_texts = '\n'.join(nearest)
    new_texts = json.dumps(return_dict, ensure_ascii = False,  separators=('\n', ':'))
    

    # Return generated text as JSON response
    return render_template('index.html', prediction_text= new_texts, initial_text = initial_texts) 
#     return jsonify({'generated_text': get_completion(payload['prompt'])})

if __name__ == '__main__':
#     app.debug = True
    app.run()
#     app.run(host='0.0.0.0')
