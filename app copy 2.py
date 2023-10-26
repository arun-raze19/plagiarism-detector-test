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
import numpy as np

#...




app = Flask(__name__)

PINECONE_INDEX_NAME = "plagiarism-checker"
DATA_FILE = "articles.csv"
NROWS = 20000

pinecone.init(      
	api_key='015a120c-a27c-4366-a563-0df742dbf89a',      
	environment='gcp-starter'      
)      
index = pinecone.Index('plagiarismdetector')

def create_model():
    model = SentenceTransformer('average_word_embeddings_komninos')

    return model

def prepare_data(data):
    # rename id column and remove unnecessary columns
    data.rename(columns={"Unnamed: 0": "article_id"}, inplace = True)
    data.drop(columns=['date'], inplace = True)

    # combine the article title and content into a single field
    data['content'] = data['content'].fillna('')
    data['content'] = data.content.swifter.apply(lambda x: ' '.join(re.split(r'(?<=[.:;])\s', x)))
    data['title_and_content'] = data['title'] + ' ' + data['content']

    # create a vector embedding based on title and article content
    encoded_articles = model.encode(data['title_and_content'], show_progress_bar=True)
    data['article_vector'] = pd.Series(encoded_articles.tolist())

    return data


def upload_items(data):
    items_to_upload = [(row.id, row.article_vector) for i, row in data.iterrows()]
    #index.upsert(items=items_to_upload, vectors=[item[1] for item in items_to_upload])
    vector_arrays = [np.array(item[1]) for item in items_to_upload]
    vectors = np.stack(vector_arrays)
    index.upsert(items=items_to_upload, vectors=vectors)
    #vectors = [np.array(item[1]) for item in items_to_upload]
    #index.upsert(items=items_to_upload, vectors=vectors)


def process_file(filename):
    data = pd.read_csv(filename, nrows=NROWS)
    data = prepare_data(data)
    upload_items(data)
    pinecone_index.info()

    return data

def map_titles(data):
    return dict(zip(uploaded_data.id, uploaded_data.title))

def map_publications(data):
    return dict(zip(uploaded_data.id, uploaded_data.publication))

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


pinecone_index = "plagiarismdetector"
model = create_model()
uploaded_data = process_file(filename=DATA_FILE)
titles_mapped = map_titles(uploaded_data)
publications_mapped = map_publications(uploaded_data)

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
