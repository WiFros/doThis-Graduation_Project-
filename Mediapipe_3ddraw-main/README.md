# Mediapipe_3ddraw
MediaPipeでハンドトラッキングを実行します。
実行して得られた手のランドマークをK3Dで可視化します。

手の動きをデジタル化して、3D表現で可視化するので様々な方向から手の動きを確認することができます。

# DEMO

![demo](https://user-images.githubusercontent.com/93971055/153741322-bccf5b26-d3c7-40b2-a208-f687a6f3d6e3.gif)


# Directory
```bash
│  demo.py
│  draw_hand.ipynb
│ 
├─result
│  └─csv
│     │ 
│     └─ landmark.csv
│          
└─utils
    └─cvfpscalc.py
```

# Requirement

mediapipe                    0.8.9.1

opencv-python                4.5.5.62   

Tensorflow                   2.3.0 or Later

tf-nightly                   2.5.0.dev or later 

jupyterlab                   3.2.8 
          
jupyterlab-widgets           1.0.2   

k3d                          2.11.0  


# Usage


```bash
git clone https://github.com/jiro-mk/Mediapipe_3ddraw.git
cd mediapipe_3ddraw
python demo.py
```


MediaPipeで取得したランドマークの可視化（jupyter notebook）

draw_hand.ipynb


# Note
MediaPipeで取得したランドマークは以下のようにcsvに保存されます。

0～20は左手、21～は右手のランドマークになります。

0_x：WRIST (Left), 0_Y：WRIST (Left), 0_Z：WRIST (Left)　… 21_x：WRIST (Right), 21_Y：WRIST (Right), 21_Z：WRIST (Right)

![image](https://user-images.githubusercontent.com/93971055/153736893-bf5fa214-3a97-4377-b0a4-fb30c2d2910e.png)

![image](https://user-images.githubusercontent.com/93971055/153736934-62549cb5-fe00-494c-a666-599715d30ff1.png)

# Author

jiro-mk
