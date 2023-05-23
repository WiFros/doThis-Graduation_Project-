# Import Libraries
from gevent import monkey
monkey.patch_all()

from flask import Flask, jsonify, request
from utilities import predict_pipeline
from gevent.pywsgi import WSGIServer
import cv2
import os
import socket
import json
import pyrebase
from Pose_Similarity_Check.pose_match import get_pose_similarity
from Pose_Similarity_Check.pose_match import draw_joints

with open("auth.json") as f:
    config = json.load(f)

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
database = firebase.database()

# file_data = dict()
# file_data["result"] = int(80) # 평균값
# file_data["result1"] = int(60)
# file_data["result2"] = int(70)
# file_data["result3"] = int(80)
# file_data["result4"] = int(90)
# file_data["result5"] = int(100)

# email_string = "example123@gmail.com"
# videoName_string = "exercise_squat_2304112205"

# database.child('temp/result/'+email_string.replace(".","_")+'/'+videoName_string).set(file_data)

# example (동영상 다운로드)
# path_on_cloud : 동영상이 저장되어있는 위치(영상이름까지 기재)
# path_on_cloud = "bjy123bjy@gmail.com/exercise/unselected/221202/VIDEO_221202_13:30_.mp4"
# path_on_cloud2 = "temp/celebrity.mp4"
# local_path : local에 동영상을 저장할 위치
# local_path = "tempDB/video/bjy123bjy.mp4"
# local_path2 = "tempDB/image/celebrity.mp4"
# 다운로드
# storage.child(path_on_cloud).download("",local_path)
# storage.child(path_on_cloud2).download("",local_path2)

# example (동영상 업로드)
# path_on_cloud : 동영상이 저장될 위치(영상이름까지 기재)
# path_on_cloud = "temp/temp.mp4"
# local_path : 올릴 동영상이 있는 위치
# local_path = "tempDB/video/temp.mp4"
# 업로드
# storage.child(path_on_cloud).put(local_path)



# Make Flask
app = Flask(__name__)

# command : <none>
@app.route('/hello',methods=['GET','POST'])
def hello():
    print("Hello!")
    print("===end===")
    return jsonify("Hello!")

# command : download_and_analyze
@app.route('/download_and_analyze',methods=['POST'])
def download():
    # Get parameters to dictionary format
    parameter_dict = request.args.to_dict()
    # test
    print('==parameter_dict==')
    print(parameter_dict)
    # If no data
    if len(parameter_dict) == 0:
        print("No parameter")
        return jsonify("ERROR 1: No parameter")

    # If have data, convert into key-value
    # parameters = ''
    keys = []
    values = []
    for key in parameter_dict.keys():
        keys.append(key)
        values.append(request.args[key])
    print('keys:', keys)
    print('values:', values)

    Squat_string = values[0].split('/')[-1].split('_')[1] # [method]만 추출
    email_string = values[0].split('/')[3]# example123@gmail.com만 추출
    videoName_string = values[0].split('/')[-1] # exercise_[method]_[timestamp]만 추출

    # Download video (user)
    path_on_cloud = values[0]+".mp4"
    local_path = "tempDB/video/"+videoName_string+".mp4"
    print('path_on_cloud:', path_on_cloud)
    print('local_path:', local_path+'\n')
    storage.child(path_on_cloud).download("",local_path)

    # Download video (pro)
    path_on_cloud_pro = "temp/video/pro/"+Squat_string+".mp4" 
    local_path_pro = "tempDB/video/"+Squat_string+".mp4"
    print('path_on_cloud (pro):', path_on_cloud)
    print('local_path (pro):', local_path+'\n')
    storage.child(path_on_cloud_pro).download("",local_path_pro)

    # Analyze
    print('====[Analyzing]====')
    results, frame_pairs, top_indices = get_pose_similarity(local_path_pro, local_path, Squat_string)

    # % 변환
    results = [arr[0][0]*100 for arr in results]

    print('results: ',results)

    # Make result contents (1. json file)
    file_data = dict()
    file_data["result"] = int(sum(results)/len(results)) # 평균값
    file_data["result1"] = int(results[0])
    file_data["result2"] = int(results[1])
    file_data["result3"] = int(results[2])
    file_data["result4"] = int(results[3])
    file_data["result5"] = int(results[4])
    # file_data["bb"] = {'x': 100, 'y': 70, 'z': 88}
    
    # Make image files (2. image files)
    for i, index in enumerate(top_indices):
        frame1, frame2 = frame_pairs[index]
        cv2.imwrite(f"tempDB/image/best{i + 1}_user_org.jpg", frame1)
        cv2.imwrite(f"tempDB/image/best{i + 1}_pro_org.jpg", frame2)
        
        # 관절 그리기
        joint_image1 = draw_joints(f"tempDB/image/best{i + 1}_user_org.jpg")
        joint_image2 = draw_joints(f"tempDB/image/best{i + 1}_pro_org.jpg")

        # 관절이 그려진 이미지 저장
        cv2.imwrite(f"tempDB/image/best{i + 1}_user.jpg", joint_image1)
        cv2.imwrite(f"tempDB/image/best{i + 1}_pro.jpg", joint_image2)

    # Make graph file (3. graph file) -> pose_match.py

    # Upload result on realtime DB
    # storage.child('temp/result/'+email_string+'/'+videoName_string+'.json').put('tempDB/result/result.json')
    database.child('temp/result/'+email_string.replace(".","_")+'/'+videoName_string).set(file_data)

    # Upload image files
    for i in range(1,6):
        cloud_path_usr = 'temp/image/'+email_string+'/'+videoName_string+'/best'+str(i)+'_user.jpg'
        cloud_path_pro = 'temp/image/'+email_string+'/'+videoName_string+'/best'+str(i)+'_pro.jpg'
        local_path_usr_ = 'tempDB/image/best'+str(i)+'_user.jpg'
        local_path_pro_ = 'tempDB/image/best'+str(i)+'_pro.jpg'
        storage.child(cloud_path_usr).put(local_path_usr_)
        storage.child(cloud_path_pro).put(local_path_pro_)

    # Upload graph file
    graph_path = "temp/image/"+email_string+"/"+videoName_string+"/graph.png"
    graph_local_path = 'tempDB/image/graph.png'
    storage.child(graph_path).put(graph_local_path)

    # Remove temporary files in the volume
    # os.remove('tempDB/result/result.json')
    for i in range(1,6):
        pass
        os.remove(f'tempDB/image/best{i}_user_org.jpg')
        os.remove(f'tempDB/image/best{i}_user.jpg')
        os.remove(f'tempDB/image/best{i}_pro_org.jpg')
        os.remove(f'tempDB/image/best{i}_pro.jpg')
    os.remove(local_path)
    os.remove(local_path_pro)
    os.remove(graph_local_path)

    # TODO: tempDB에 저장되는 파일들의 unique성이 필요함. 다중 접속 시, 파일명이 겹침.

    # TODO: firebase에 저장되어있는 영상 삭제되도록 해야 함.
    # fileName = 'temp/exercise/'+values[0].split('/')[-1]+'.mp4'
    # storage.delete(fileName,token=None)


    # Jobs finished!
    return jsonify("RESULT")

# command : clean
@app.route('/clean',methods=['DELETE'])
def clean():
    return jsonify("Clean Complete")


# Run Flask server
if __name__ == "__main__":
    thisIP = socket.gethostbyname(socket.gethostname())
    print("This PC IP:", thisIP)
    # app.run(host=thisIP, debug=True)
    # app.run(host='localhost', debug=True)
    http_server = WSGIServer((thisIP,5000),app)
    http_server.serve_forever()
