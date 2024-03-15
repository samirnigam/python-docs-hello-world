from google.cloud import storage
from pathlib import Path
from flask import Flask
app = Flask(__name__)

file_list=[]

@app.route("/")
def hello():
    #return "Hello, World!"
    path_to_private_key = 'fifth-compass-415612-76f634511b19.json'
    client = storage.Client.from_service_account_json(json_credentials_path=path_to_private_key)
    bucket = storage.Bucket(client, 'hackathon1415')
    
    str_folder_name_on_gcs = 'JD/'
    
    # Create the directory locally
    Path(str_folder_name_on_gcs).mkdir(parents=True, exist_ok=True)
    blobs = bucket.list_blobs(prefix=str_folder_name_on_gcs)
    for blob in blobs:
        if not blob.name.endswith('/'):
            # This blob is not a directory!
            if blob.name.endswith('docx'):
                 file_list.append(blob.name)
            #print(f'Downloading file [{blob.name}]')
            #blob.download_to_filename(f'./{blob.name}')
    return file_list
