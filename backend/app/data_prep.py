import sqlite3
import csv
import numpy as np
from scipy.spatial.distance import cosine
from utils import number_sent
import torch
from transformers import AutoTokenizer, AutoModel

def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    
    scores = model_output.pooler_output
    # print(scores)
    return embeddings[0].cpu().numpy().tolist()

    

def search_content(number_sent, input_vector):

    conn = sqlite3.connect('text_db.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM text_db")
    rows = cursor.fetchall()
   
    distances = []
    for row in rows:
        vector = [float(x) for x in row[1].split(',')]
        distance = cosine(input_vector, vector)
        distances.append((distance, row[1], row[0])) 
        
    distances.sort(key=lambda x: x[0])

    conn.close()
    out = []
    for i in range(number_sent):
        out.append(distances[i][2])
        
    return out

