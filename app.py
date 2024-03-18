import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import json
from operator import itemgetter
from urllib.parse import urlparse
from azure.storage.blob import ContainerClient, BlobClient
from google.cloud import storage
from pathlib import Path
from pypdf import PdfReader
import re
import textract
import os

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Home"

@app.route("/get-user/<user_id>")
def get_user(user_id):
    user_data = {
        "user_id": user_id,
        "name": "Rahul sur",
        "email": "rahulsur.work@gmail.com"
    }

    extra = request.args.get("extra")
    if extra:
        user_data["extra"] = extra

    return jsonify(user_data), 200

@app.route("/create-user", methods=["POST"])
def create_user():
    if request.method == "POST":
        data = request.get_json()
        return jsonify(data), 201

@app.route("/job-description", methods=["POST"])
def job_description():
    #---gcp-----
    def text_extract_doc(file_path):
        #print("before textract.process" + file_path)
        text = textract.process(file_path)
        #print("before text.decode")
        text = text.decode()
        #print("after text.decode")
        ##text = clean_text(text)
        return text

    def text_extract_pdf(file_path):
        ### Reading from PDF version of CV
        reader = PdfReader(file_path)
        list_page_pbj_mod = reader.pages

        ### Converting pdf to text
        text = ""
        for page in list_page_pbj_mod:
            text += page.extract_text() + "\n"
        #text = clean_text(text)
        return text

    def convert_to_text(file_path):
        ext = os.path.splitext(file_path)[-1].lower()
        # Now we can simply use == to check for equality, no need for wildcards.
        if ext == ".pdf":
            return text_extract_pdf(file_path)
        elif ext == ".docx":
            return text_extract_doc(file_path)
        else:
            print( "is an unknown file format." )

        return ""   

    #---gcp-----
    result = []
    ntop = 0
    confidenceScore = 0
    response = {
        "status": "success",
        "count": ntop,
        "metadata": {
            "confidenceScore": confidenceScore
        },
        "results": None
    }      
    try:
        data = request.get_json()
        path_to_private_key = 'fifth-compass-415612-76f634511b19.json'
        client = storage.Client.from_service_account_json(json_credentials_path=path_to_private_key)
        inputPath = data["inputPath"]
        path_arr = inputPath.split("/")
        idx = path_arr.index("browser")
        n = len(path_arr)
        #bucket = storage.Bucket(client, 'hackathon1415')
        #bucket = storage.Bucket(client, 'hackathontestdata2024')
        print(path_arr[idx + 1])
        bucket = storage.Bucket(client, path_arr[idx + 1])
        str_folder_name_on_gcs = ''
        ntop = data["noOfMatches"]
        confidenceScore = data["threshold"]
        response = {
            "status": "success",
            "count": ntop,
            "metadata": {
                "confidenceScore": confidenceScore
            },
            "results": None
        }        
        # Initialize NLTK stopwords
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

        # Initialize OpenAI API
        openai.api_key = 'ded067ba7c36469184fc9aa6601c12a7'

        def preprocess_text(text):
            # Tokenization and removal of stopwords
            tokens = nltk.word_tokenize(text.lower())
            tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
            return ' '.join(tokens)

        def calculate_similarity(job_description, resume):
            # Preprocess job description and resume
            processed_job_description = preprocess_text(job_description)
            processed_resume = preprocess_text(resume)
            
            # Calculate TF-IDF vectors
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([processed_job_description, processed_resume])
            
            # Calculate cosine similarity between job description and resume
            similarity_score = cosine_similarity(tfidf_matrix)[0, 1]
            return similarity_score

        def match_resume_to_job(job_description, resume):
            # Calculate similarity score
            similarity_score = calculate_similarity(job_description, resume)
            
            # Optionally, use OpenAI API for additional processing or scoring
            
            return similarity_score

        # Sample job description and resume
        job_description = data["context"]

        resumeList = []

        #begin ----gcp------
        # Create the directory locally
        index = 0
        blobs = bucket.list_blobs(prefix=str_folder_name_on_gcs)
        for blob in blobs:
            if not blob.name.endswith('/'):
                if blob.name.endswith('.pdf') or blob.name.endswith('.docx'):
                    # This blob is not a directory!
                    print(f'Downloading file [{blob.name}]')
                    index +=1
                    try:
                        resume_record = {
                            "id": "",
                            "path": "",
                            "content": ""
                        }        
                        blob.download_to_filename(f'./{blob.name}')
                        text = convert_to_text(f"./{blob.name}")
                        if len(text) > 0:
                            pathname = blob.name.split('.')[0].strip()
                            #print(pathname)
                            idname = pathname.split('/')[-1].strip()
                            #print(idname)
                            resume_record["id"] = idname
                            resume_record["path"] = blob.name
                            resume_record["content"] = text
                            #insert into resume list
                            resumeList.append(resume_record)
                    except:
                        text=""
                        print("error parsing blob")
                    #print(text)    

                    # if index > 100:
                    #     break
        #end ----gcp------
        # Match job description to resume and get similarity score

        for cv in resumeList:
            item = {
                "id": "",
                "path": "",
                "score": 0
            }

            similarity_score = match_resume_to_job(job_description, cv["content"])
            #calc_score = similarity_score
            calc_score = confidenceScore + similarity_score
            print(cv["id"])
            item["id"] = cv["id"]
            item["path"] = cv["path"]
            item["score"] = calc_score
            #print(item["path"])
            result.append(item)
            #print(f"Similarity score between job description and resume: {similarity_score:.2f}")
            print(f"Similarity score between job description and resume: {calc_score:.2f}")

        finalResult = sorted(result, key=itemgetter('score'), reverse=True)[:ntop]
        response["results"] = finalResult
        #print(final)
    except BaseException as e:
        print("Something went wrong:" + str(e))
        response["status"] = "error"
    finally:
        print("The 'try except' is finished")    

    return jsonify(response), 201

if __name__ == "__main__":
    app.run(debug=True)
