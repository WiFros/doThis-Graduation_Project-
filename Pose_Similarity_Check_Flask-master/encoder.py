import subprocess

def change_fps(video_path, output_path, fps):
    cmd = ['ffmpeg', '-i', video_path, '-r', str(fps), output_path]
    subprocess.run(cmd, check=True)

video_path1 = 'pushups-sample_exp.mp4'
video_path2 = 'pushups-sample_user.mp4'

output_path1 = 'output1.mp4'
output_path2 = 'output2.mp4'

change_fps(video_path1, output_path1, 30)
change_fps(video_path2, output_path2, 30)
