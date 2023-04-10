import matplotlib as plt

def plot_line(a, b):
    # 관절마다 선긋기
    if (a.any()> 0 and b.any()>0): plt.plot([a[0], b[0]], [a[1], b[1]], 'k-')
        
def draw_skeleton(sample, pattern):
    
    keypoint = ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder',
                'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle']
    skeleton = sample
    
    # 각 신체부위의 xy좌표 추출
    Nose = skeleton[[0,17]]
    LEye = skeleton[[1,18]]
    REye = skeleton[[2,19]]
    LEar = skeleton[[3,20]]
    REar = skeleton[[4,21]]
    LShoulder = skeleton[[5,22]]
    RShoulder = skeleton[[6,23]]
    LElbow = skeleton[[7,24]]
    RElbow = skeleton[[8,25]]
    LWrist = skeleton[[9,26]]
    RWrist = skeleton[[10,27]]
    LHip = skeleton[[11,28]]
    RHip = skeleton[[12,29]]
    LKnee = skeleton[[13,30]]
    RKnee = skeleton[[14,31]]
    LAnkle = skeleton[[15,32]]
    RAnkle = skeleton[[16,33]]
    
    inbox={'facecolor':'w','edgecolor':'r','boxstyle':'round','alpha':0.5}
    outbox={'facecolor':'w','edgecolor':'b','boxstyle':'round','alpha':0.5}
    
    # 각 관절 점 및 텍스트 표시
    for idx in range(len(sample)//2):
        plt.plot(sample[idx], sample[17+idx], pattern)
        plt.text(sample[idx], sample[17+idx], keypoint[idx], verticalalignment='bottom' , horizontalalignment='center' )
    
    # 신체에 맞게 각 관절 선으로 연결
    plot_line(LEye, Nose)
    plot_line(REar, REye)
    plot_line(REye, Nose)
    plot_line(LEar, LEye)
    plot_line(LShoulder, LElbow)
    plot_line(LElbow, LWrist)
    plot_line(RShoulder, RElbow)
    plot_line(RElbow, RWrist)
    plot_line(LHip, LKnee)
    plot_line(LKnee, LAnkle)
    plot_line(RKnee, RAnkle)
    plot_line(RHip, RKnee)
    plot_line(LHip, LShoulder)
    plot_line(RHip, RShoulder)
    plot_line(RHip, LHip)
    plot_line(LShoulder, RShoulder)
    plot_line(LShoulder, Nose)
    plot_line(RShoulder, Nose)
    
def skeleton_plot(sample, color):
    # 그래프 padding 값
    pad_ori = 40
    plt.subplot(122)
    plt.title('skeleton')
    X_ori = sample
    
    # 관절의 최대 최소값을 구해 그래프 영역 설정
    x_max = max(X_ori[:17]) + pad_ori
    x_min = min(i for i in X_ori[:17] if i > 0) - pad_ori
    y_max = max(X_ori[17:]) + pad_ori
    y_min = min(j for j in X_ori[17:] if j > 0) - pad_ori
    plt.xlim(x_min, x_max)
    plt.ylim(y_max, y_min)
    draw_skeleton(X_ori, color)