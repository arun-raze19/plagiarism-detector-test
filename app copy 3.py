import numpy as np
from dotenv import load_dotenv
from flask import Flask
from flask import render_template
from flask import request
from flask import url_for
import json
import os
import pandas as pd
import pinecone
import re
import requests
from sentence_transformers import SentenceTransformer
from statistics import mean
import swifter

app = Flask(__name__)

PINECONE_INDEX_NAME = "plagiarism-checker"
DATA_FILE = "articles.csv"
NROWS = 20000

pinecone.init(api_key="015a120c-a27c-4366-a563-0df742dbf89a", environment="gcp-starter")

# Initialize Pinecone index
pinecone_index = pinecone.Index("plagiarismdetector")


def create_model():
  model = SentenceTransformer("average_word_embeddings_komninos")
  return model
  
    


def prepare_data(data):
  encoded_articles = []
  for content in data["title_and_content"]:
    vector = model.encode(content)
    encoded_articles.append(np.array(vector))
    
        
  
    
    # Encode vectors as numpy arrays
    

  data["article_vector"] = encoded_articles

    # Other data processing
  data.rename(columns={"Unnamed: 0": "article_id"}, inplace=True)
  data.drop(columns=["date"], inplace=True)

  data["content"] = data["content"].fillna("")
  data["content"] = data.content.swifter.apply(
      lambda x: " ".join(re.split(r"(?<=[.:;])\s", x))
    )
  data["title_and_content"] = data["title"] + " " + data["content"]

  return data


def upload_items(data):
  
    # Stack vectors into array
  vectors = [row.article_vector for _, row in data.iterrows()]
  stacked_vectors = np.stack(vectors)

  items_to_upload = [(row.id, row.article_vector) for _, row in data.iterrows()]

  pinecone_index.upsert(items=items_to_upload, vectors=stacked_vectors)

def process_file(filename):
  data = pd.read_csv(filename, nrows=NROWS)
  data = prepare_data(data)
  upload_items(data)

  return data


# Other functions...

model = create_model()
uploaded_data = process_file(DATA_FILE)

# Map titles/publications after processing
titles_mapped = dict(zip(uploaded_data.id, uploaded_data.title))
publications_mapped = dict(zip(uploaded_data.id, uploaded_data.publication))

    
      

# Flask app...
# Other functions

def map_titles(data):
  return dict(zip(data.id, data.title))

def map_publications(data):
  return dict(zip(data.id, data.publication))  

def query_pinecone(originalContent):

  query_content = str(originalContent)
  query_vectors = [model.encode(query_content)]

  query_results = pinecone_index.query(queries=query_vectors, top_k=10)
  res = query_results[0]

  results_list = []
  for idx, _id in enumerate(res.ids):
    results_list.append({
      "id": _id,
      "title": titles_mapped[int(_id)], 
      "publication": publications_mapped[int(_id)],
      "score": res.scores[idx],
    })

  return json.dumps(results_list)

# Flask app

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/api/search", methods=["POST", "GET"]) 
def search():
  if request.method == "POST":
    return query_pinecone(request.form.get("originalContent", ""))
  if request.method == "GET":
    return query_pinecone(request.args.get("originalContent", ""))

  return "Only GET and POST methods are allowed for this endpoint"