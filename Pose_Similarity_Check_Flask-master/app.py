from flask import Flask, json, request
from modules import similarity_check as sc

app = Flask(__name__)

@app.route('/')
def root():
    return 'welcome to flask'

@app.route('/data/get')
def data_get():
    return json.jsonify({'success': '성공'}), 200

@app.route('/post', methods=['POST'])
def post():
    pose = request.get_json()
    pose1 = pose[0]
    pose2 = pose[3]
    print(type(pose1))
    similarity = sc.get_pose_similarity(pose1, [4032, 3024, 3], pose2, [4032, 3024, 3])
    print(similarity)
    return str(similarity)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
