import firebase_admin
from firebase_admin import db
from firebase_admin import credentials
from firebase_admin import storage
import os
import json

def process_map(map_name, map_json, visualize=False):
    json_blob = bucket.get_blob(map_json) # adapt to download files i want
    if json_blob is not None:
        json_data = json_blob.download_as_string()
        x = json.loads(json_data)
        print(x)

        # make own folder to upload data after modifying x
        # make a new file name and then upload
    else:
        print("map file was missing")


def unprocessed_maps_callback(m):
    if type(m.data) == str:
        # a single new map just got added
        process_map(m.path.lstrip('/'), m.data)
    elif type(m.data) == dict:
        # this will be a dictionary of all the data that is there initially
        for map_name, map_json in m.data.items():
            process_map(map_name, map_json)
    
# Fetch the service account key JSON file contents
cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))

# Initialize the app with a service account, granting admin privileges
app = firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://invisible-map-sandbox.firebaseio.com/',
    'storageBucket': 'invisible-map.appspot.com'
})

ref = db.reference('/unprocessed_maps')
to_process = ref.get()
bucket = storage.bucket(app=app)

ref.listen(unprocessed_maps_callback)

