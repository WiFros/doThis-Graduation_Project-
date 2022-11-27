import json

with open('video1.json') as json_file:
    json_data = json.load(json_file)
    print(json_data[0]['Pose'])
