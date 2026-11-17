# coding: utf-8
"""
カメラキャプチャ＆基本的な画像処理サンプル

Webカメラから画像を1フレームずつ取り込み、キー操作で選んだ処理を施して表示します。

使い方:
    python capture.py            # カメラ番号 0 を使う
    python capture.py 1          # カメラ番号 1 を使う
    python capture.py movie.mp4  # カメラの代わりに動画ファイルを使う（繰り返し再生）

キー操作は起動時にターミナルへ表示されます（README.md も参照）。

自分で処理を追加したいときは、下の「proc_custom」関数を書き換えて
キー 'u' を押してください。
"""

import os
import sys
import time
import random

import cv2
import numpy as np


# ----------------------------------------------------------------------
# 設定
# ----------------------------------------------------------------------
PROC_SCALE = 0.5        # 処理速度のため、入力画像をこの倍率に縮小してから処理する
WINDOW_NAME = "OpenCV Capture"
SHOW_HELP_AT_START = True   # 起動時にウィンドウにヘルプを重ねて表示する（? キーで ON/OFF）
HELP_DIM = 0.3              # ヘルプ表示中の背景の明るさ（0〜1、小さいほど暗い）

# 学習済みモデルの置き場所（このスクリプトと同じフォルダの models/）
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HAAR_DIR = os.path.join(MODEL_DIR, "haarcascades")
YUNET_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")


# ----------------------------------------------------------------------
# 状態（パラメータ等）をまとめて持つクラス
# ----------------------------------------------------------------------
class State:
    def __init__(self, proc_shape):
        h, w = proc_shape[:2]

        # 実行中の処理名
        self.proc = "color"

        # 前処理（ぼかし）の種類とパラメータ
        self.blur_proc = "gaussian"
        self.reset_param()

        # 前フレーム（オプティカルフロー用）
        self.prvs = None

        # HSV 表示用
        self.hsv = np.zeros((h, w, 3), np.uint8)
        self.hsv[..., 1] = 255

        # ラベリング用のランダムカラー表（HSV）
        colors = [[0, 0, 0]]
        for _ in range(1, 256):
            colors.append([random.randint(0, 180), random.randint(120, 255), 255])
        self.colors_hsv = np.array(colors, dtype=np.uint8)

        # 顔検出器は必要になったときに読み込む（load_haar / load_yunet）
        self.haar = None          # Haar-like 特徴カスケード（古典的手法）
        self.yunet = None         # YuNet（CNN による顔検出）
        self.pose = None          # MediaPipe Pose（人体の骨格推定）
        self.profile_face = True

        # Lucas-Kanade 追跡用
        self.lk_fnum = 1000
        # Shi-Tomasi コーナー検出器のためのパラメータ
        self.feature_params = dict(maxCorners=self.lk_fnum,
                                   qualityLevel=0.001,
                                   minDistance=7,
                                   blockSize=7)
        # Lucas-Kanade 法によるオプティカルフローのためのパラメータ
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.lk_color = np.random.randint(0, 255, (self.lk_fnum, 3))
        self.lk_reset = True
        self.lk_points = None
        self.track_list = []

        # SIFT / CLAHE は必要になったときに作る
        self.sift = None
        self.clahe = None

    # --- ぼかしパラメータ（↑↓キーで変更、ESCでリセット）---
    def increase_param(self):
        self.gaussian_scale = min(self.gaussian_scale + 6, 149)
        self.median_scale = min(self.median_scale + 6, 199)
        self.bilateral_scale = min(self.bilateral_scale + 5, 30)

    def decrease_param(self):
        self.gaussian_scale = max(self.gaussian_scale - 6, 1)
        self.median_scale = max(self.median_scale - 6, 1)
        self.bilateral_scale = max(self.bilateral_scale - 5, 1)

    def reset_param(self):
        # ガウシアン・メディアンのカーネルサイズは奇数である必要がある
        self.gaussian_scale = 1
        self.median_scale = 1
        self.bilateral_scale = 1


# ----------------------------------------------------------------------
# 顔検出器の読み込み
# ----------------------------------------------------------------------
def load_haar(st):
    """Haar-like 特徴カスケードを読み込む。使えなければ False を返す"""
    if st.haar is not None:
        return st.haar is not False
    if not hasattr(cv2, "CascadeClassifier"):
        # OpenCV 5 では CascadeClassifier は contrib 版にのみ含まれる
        print("この OpenCV (%s) には CascadeClassifier がありません。" % cv2.__version__)
        print("opencv-contrib-python を入れてください（README.md 参照）。")
        st.haar = False
        return False
    names = {"face": "haarcascade_frontalface_default.xml",
             "eye": "haarcascade_eye_tree_eyeglasses.xml",
             "body": "haarcascade_upperbody.xml",
             "profile": "haarcascade_profileface.xml"}
    st.haar = {}
    for key, fname in names.items():
        c = cv2.CascadeClassifier(os.path.join(HAAR_DIR, fname))
        if c.empty():
            print("カスケードファイルを読み込めません: %s" % os.path.join(HAAR_DIR, fname))
            st.haar = False
            return False
        st.haar[key] = c
    return True


def load_yunet(st, size):
    """YuNet 顔検出器を読み込む。使えなければ False を返す"""
    if st.yunet is None:
        if not hasattr(cv2, "FaceDetectorYN"):
            print("この OpenCV (%s) には FaceDetectorYN がありません（4.8 以降が必要）。" % cv2.__version__)
            st.yunet = False
        elif not os.path.exists(YUNET_MODEL):
            print("YuNet のモデルファイルがありません: %s" % YUNET_MODEL)
            st.yunet = False
        else:
            # 引数: モデル, 設定(不要), 入力サイズ, スコア閾値, NMS閾値, 最大候補数
            st.yunet = cv2.FaceDetectorYN.create(YUNET_MODEL, "", size, 0.7, 0.3, 5000)
    if st.yunet is False:
        return False
    st.yunet.setInputSize(size)
    return True


def show_unavailable(img, msg):
    result = img.copy()
    cv2.putText(result, msg, (10, result.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2)
    return result


# ----------------------------------------------------------------------
# 前処理（ぼかし）
# ----------------------------------------------------------------------
def blur(img, st):
    if st.blur_proc == "gaussian":
        k = st.gaussian_scale
        return cv2.GaussianBlur(img, (k, k), 0)
    elif st.blur_proc == "median":
        return cv2.medianBlur(img, st.median_scale)
    elif st.blur_proc == "bilateral":
        return cv2.bilateralFilter(img, st.bilateral_scale, 75, 75)
    return img


# ----------------------------------------------------------------------
# 各種画像処理
#   引数 img: BGR カラー画像 (uint8)、st: State
#   戻り値: 表示する画像（カラーでもグレースケールでもよい）
# ----------------------------------------------------------------------
def proc_color(img, st):
    return img


def proc_gray(img, st):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _sobel(img, dx, dy, ksize=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)


def proc_sobelx(img, st):
    dx = _sobel(img, 1, 0)
    return np.clip(np.abs(dx * 10), 0, 255).astype(np.uint8)


def proc_sobely(img, st):
    dy = _sobel(img, 0, 1)
    return np.clip(np.abs(dy * 10), 0, 255).astype(np.uint8)


def proc_sobel(img, st):
    dx = _sobel(img, 1, 0)
    dy = _sobel(img, 0, 1)
    mag = np.sqrt((dx * dx + dy * dy) / 2) * 10
    return np.clip(mag, 0, 255).astype(np.uint8)


def proc_angle(img, st):
    # 勾配の向きを色相(H)、強さを明度(V)で表示
    dx = _sobel(img, 1, 0, ksize=3)
    dy = _sobel(img, 0, 1, ksize=3)
    ang = np.mod(np.arctan2(dy, dx), 2 * np.pi)          # 0 〜 2π
    hsv = np.empty(img.shape, np.uint8)
    hsv[..., 0] = (ang / (2 * np.pi) * 180).astype(np.uint8)  # OpenCV の H は 0〜179
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(np.sqrt((dx * dx + dy * dy) / 2) * 10, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def proc_hue(img, st):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def proc_binary(img, st):
    # 大津の方法による二値化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result


def proc_sift(img, st):
    if st.sift is None:
        st.sift = cv2.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp = st.sift.detect(gray, None)
    return cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def proc_denseflow(img, st):
    # Farneback 法による密なオプティカルフロー
    nxt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if st.prvs is None or st.prvs.shape != nxt.shape:
        st.prvs = nxt
    flow = cv2.calcOpticalFlowFarneback(st.prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # 向きを色相、大きさを明度で表示
    st.hsv[..., 0] = ang * 180 / np.pi / 2
    st.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(st.hsv, cv2.COLOR_HSV2BGR)
    result = cv2.addWeighted(img, 0.5, rgb, 0.5, 0)
    # 間引いた位置に矢印（線分）を描く
    for i in range(0, result.shape[0], 15):
        for j in range(0, result.shape[1], 15):
            fx, fy = flow[i, j]
            cv2.line(result, (j, i), (int(j + fx * 3), int(i + fy * 3)), (255, 255, 255), 1)
    st.prvs = nxt
    return result


def proc_labeling(img, st):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, label = cv2.connectedComponents(binimg)
    return cv2.cvtColor(st.colors_hsv[label % 256], cv2.COLOR_HSV2BGR)


def proc_labeling2(img, st):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num, label, stats, centroids = cv2.connectedComponentsWithStats(binimg)
    result = cv2.cvtColor(st.colors_hsv[label % 256], cv2.COLOR_HSV2BGR)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 100:
            continue
        x1, y1 = x + w, y + h
        cv2.rectangle(result, (x, y), (x1, y1), (0, 0, 255))
        cv2.putText(result, "ID: " + str(i), (x, y1 + 15), font, 1, (0, 255, 255))
        cv2.putText(result, "S: " + str(area), (x, y1 + 30), font, 1, (0, 255, 255))
        cv2.putText(result, "X: " + str(int(centroids[i][0])), (x1 - 10, y1 + 15), font, 1, (0, 255, 255))
        cv2.putText(result, "Y: " + str(int(centroids[i][1])), (x1 - 10, y1 + 30), font, 1, (0, 255, 255))
    return result


def proc_facedetect(img, st):
    # Viola-Jones 法（Haar-like 特徴 + AdaBoost + カスケード）による顔検出
    if not load_haar(st):
        return show_unavailable(img, "Haar cascade not available")
    result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 正面顔（青）と目（緑）
    faces = st.haar["face"].detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = result[y:y + h, x:x + w]
        eyes = st.haar["eye"].detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    # 横顔（赤）: 学習データは片向きのみなので左右反転画像でも検出する
    if st.profile_face:
        W = gray.shape[1]
        for (x, y, w, h) in st.haar["profile"].detectMultiScale(gray, 1.3, 5):
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        flipped = cv2.flip(gray, 1)
        for (x, y, w, h) in st.haar["profile"].detectMultiScale(flipped, 1.3, 5):
            cv2.rectangle(result, (W - 1 - x - w, y), (W - 1 - x, y + h), (0, 0, 255), 2)
    return result


def proc_bodydetect(img, st):
    if not load_haar(st):
        return show_unavailable(img, "Haar cascade not available")
    result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in st.haar["body"].detectMultiScale(gray, 1.3, 5):
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return result


def proc_yunet(img, st):
    # YuNet: 小さな CNN による顔検出（OpenCV 4.8 以降 / 5 の本体に含まれる）
    h, w = img.shape[:2]
    if not load_yunet(st, (w, h)):
        return show_unavailable(img, "YuNet not available")
    result = img.copy()
    _, faces = st.yunet.detect(img)
    if faces is None:
        return result
    # faces の各行: x, y, w, h, 右目x, 右目y, 左目x, 左目y, 鼻x, 鼻y, 口右x, 口右y, 口左x, 口左y, スコア
    landmark_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
    for f in faces:
        x, y, fw, fh = f[:4].astype(int)
        cv2.rectangle(result, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        for i in range(5):
            px, py = f[4 + 2 * i:6 + 2 * i].astype(int)
            cv2.circle(result, (px, py), 3, landmark_colors[i], -1)
        cv2.putText(result, "%.2f" % f[14], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
    return result


def proc_lktracking(img, st):
    # Lucas-Kanade 法による特徴点追跡
    nxt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if st.lk_reset or st.lk_points is None or len(st.lk_points) == 0:
        st.prvs = nxt
        st.track_list = []
        st.lk_points = cv2.goodFeaturesToTrack(nxt, mask=None, **st.feature_params)
        st.lk_reset = False
        if st.lk_points is None:       # 特徴点が見つからなかった
            return img
    p0 = st.lk_points.reshape(-1, 1, 2)

    # オプティカルフローを計算
    p1, status, err = cv2.calcOpticalFlowPyrLK(st.prvs, nxt, p0, None, **st.lk_params)

    # 追跡に成功した点だけ残す
    good_new = p1[status == 1]
    good_old = p0[status == 1]

    # 軌跡を描画
    track = np.zeros_like(img)
    result = img.copy()
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        color = st.lk_color[i].tolist()
        cv2.line(track, (int(a), int(b)), (int(c), int(d)), color, 2)
        cv2.circle(result, (int(a), int(b)), 5, color, -1)

    st.track_list.append(track)
    if len(st.track_list) > 20:
        st.track_list.pop(0)
    for t in st.track_list:
        result = np.where(t != 0, t, result)

    st.lk_points = good_new
    st.prvs = nxt
    return result


def proc_dog(img, st):
    # DoG (Difference of Gaussian) フィルタ: ↑↓キーでガウシアンの大きさを変更
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = st.gaussian_scale * 2 + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    return np.clip((gray - blurred) * 3.0 + 128.0, 0, 255).astype(np.uint8)


def proc_hist_eq(img, st):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def proc_hist_eq_clahe(img, st):
    if st.clahe is None:
        st.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return st.clahe.apply(gray)


# ----------------------------------------------------------------------
# 人体の骨格推定（中身は pose_estimation.py を参照）
# ----------------------------------------------------------------------
def load_pose(st):
    """骨格推定器を読み込む。使えなければ False を返す"""
    if st.pose is None:
        import pose_estimation
        missing = [p for p in (pose_estimation.PERSON_DET_MODEL, pose_estimation.POSE_MODEL)
                   if not os.path.exists(p)]
        if missing:
            print("骨格推定のモデルがありません: %s" % ", ".join(os.path.basename(p) for p in missing))
            print("python download_models.py でダウンロードしてください。")
            st.pose = False
        else:
            st.pose = pose_estimation.PoseEstimator()
    return st.pose is not False


def proc_pose(img, st):
    # 人体の骨格推定（33 点）。k キーで時間方向の平滑化を ON/OFF
    if not load_pose(st):
        return show_unavailable(img, "Pose model not found: python download_models.py")
    import pose_estimation
    result = img.copy()
    res = st.pose.process(img)
    if res is not None:
        pose_estimation.draw_pose(result, *res)
    return result


def proc_custom(img, st):
    """
    ★ ここに自分の処理を書いてみよう（キー 'u' で実行）★
    例として、Canny エッジ検出を行っています。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges


# ----------------------------------------------------------------------
# キーと処理の対応表   キー: (処理名, 関数, 説明)
# ----------------------------------------------------------------------
PROCS = {
    'c': ("color",         proc_color,         "カラー（そのまま）"),
    'g': ("gray",          proc_gray,          "グレースケール"),
    'x': ("sobelx",        proc_sobelx,        "Sobel フィルタ（x方向）"),
    'y': ("sobely",        proc_sobely,        "Sobel フィルタ（y方向）"),
    's': ("sobel",         proc_sobel,         "勾配強度"),
    'a': ("angle",         proc_angle,         "勾配方向（色相）と強度（明度）"),
    'h': ("hue",           proc_hue,           "色相のみ表示"),
    'b': ("binary",        proc_binary,        "二値化（大津の方法）"),
    ':': ("sift",          proc_sift,          "SIFT 特徴点"),
    'f': ("denseflow",     proc_denseflow,     "密なオプティカルフロー（Farneback）"),
    'l': ("labeling",      proc_labeling,      "ラベリング"),
    ';': ("labeling2",     proc_labeling2,     "ラベリング＋外接矩形・重心"),
    't': ("lktracking",    proc_lktracking,    "特徴点追跡（Lucas-Kanade）"),
    'F': ("facedetect",    proc_facedetect,    "顔検出（Haar-like 特徴・Viola-Jones 法）"),
    'Y': ("yunet",         proc_yunet,         "顔検出（YuNet・CNN）"),
    'K': ("pose",          proc_pose,          "人体の骨格推定（MediaPipe Pose）"),
    'U': ("bodydetect",    proc_bodydetect,    "上半身検出（Haar-like）"),
    'd': ("dog",           proc_dog,           "DoG フィルタ"),
    'e': ("hist_eq",       proc_hist_eq,       "ヒストグラム平坦化"),
    'E': ("hist_eq_CLAHE", proc_hist_eq_clahe, "適応的ヒストグラム平坦化（CLAHE）"),
    'u': ("custom",        proc_custom,        "自作の処理（proc_custom を編集）"),
}
PROC_FUNCS = {name: func for (name, func, _) in PROCS.values()}

BLURS = {
    'G': ("gaussian",  "前処理: ガウシアンぼかし"),
    'm': ("median",    "前処理: メディアンフィルタ"),
    'B': ("bilateral", "前処理: バイラテラルフィルタ"),
}


def print_help():
    print("-" * 60)
    print("キー操作（画像ウィンドウを選択した状態で押してください）")
    for k, (_, _, desc) in PROCS.items():
        print("  %s : %s" % (k, desc))
    for k, (_, desc) in BLURS.items():
        print("  %s : %s" % (k, desc))
    print("  ↑ / + : ぼかしを強く        ↓ / - : ぼかしを弱く")
    print("  ESC   : ぼかしをリセット")
    print("  p     : 顔検出中に横顔検出の ON/OFF")
    print("  k     : 骨格推定中に平滑化の ON/OFF")
    print("  n     : NDI 送信の ON/OFF（ndi-python が必要・任意）")
    print("  ?     : ヘルプ表示の ON/OFF（ウィンドウ）")
    print("  q     : 終了")
    print("-" * 60)


# ----------------------------------------------------------------------
# ウィンドウへのヘルプ表示（? キーで ON/OFF）
#   OpenCV 5 以降は内蔵フォントで日本語を描けるので日本語で、
#   OpenCV 4 では日本語が描けない（? になる）ので英語で表示する。
# ----------------------------------------------------------------------
HELP_EN = {   # OpenCV 4 用の英語の説明（ない処理は処理名をそのまま表示）
    "color": "Color (as is)", "gray": "Grayscale",
    "sobelx": "Sobel filter (x)", "sobely": "Sobel filter (y)",
    "sobel": "Gradient magnitude", "angle": "Gradient direction (hue) & magnitude",
    "hue": "Hue only", "binary": "Binarize (Otsu)", "sift": "SIFT keypoints",
    "denseflow": "Dense optical flow (Farneback)", "labeling": "Labeling",
    "labeling2": "Labeling + bounding box & centroid",
    "lktracking": "Feature tracking (Lucas-Kanade)",
    "facedetect": "Face detection (Haar-like, Viola-Jones)",
    "yunet": "Face detection (YuNet, CNN)", "bodydetect": "Upper body detection (Haar-like)",
    "pose": "Human pose estimation (MediaPipe Pose)",
    "dog": "DoG filter", "hist_eq": "Histogram equalization",
    "hist_eq_CLAHE": "Adaptive histogram equalization (CLAHE)",
    "custom": "Your own processing (edit proc_custom)",
    "gaussian": "Gaussian blur", "median": "Median filter", "bilateral": "Bilateral filter",
}

COLOR_HEAD = (255, 255, 0)      # 見出し（水色）
COLOR_ITEM = (255, 255, 255)    # 通常（白）
COLOR_CURRENT = (0, 255, 255)   # 選択中の処理（黄色）


class TextDrawer:
    """OpenCV 4 / 5 の違いを吸収して文字を描くクラス（px は文字の大きさ[画素]）"""

    def __init__(self):
        self.face = None
        if hasattr(cv2, "FontFace"):        # OpenCV 5 以降
            try:
                self.face = cv2.FontFace("sans")
            except cv2.error:
                self.face = None
        self.japanese = self.face is not None

    def _hershey_scale(self, px):
        return cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, max(int(px * 0.7), 1), 1)

    def width(self, text, px):
        if self.face is not None:
            return cv2.getTextSize((0, 0), text, (0, 0), self.face, int(px))[2]
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self._hershey_scale(px), 1)
        return w

    def put(self, img, text, org, color, px, bold=False):
        org = (int(org[0]), int(org[1]))
        if self.face is not None:
            cv2.putText(img, text, org, color, self.face, int(px), 700 if bold else 400)
        else:
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, self._hershey_scale(px),
                        color, 2 if bold else 1, cv2.LINE_AA)


def help_columns(st, ja):
    """ヘルプの内容を (キー, 説明, 色) のリスト2列分として返す"""
    def desc_of(name, desc):
        return desc if ja else HELP_EN.get(name, name)

    col1 = [("", "処理の切り替え" if ja else "Processing", COLOR_HEAD)]
    for k, (name, _, desc) in PROCS.items():
        col1.append((k, desc_of(name, desc), COLOR_CURRENT if name == st.proc else COLOR_ITEM))

    col2 = [("", "前処理（ぼかし）" if ja else "Pre-filter (blur)", COLOR_HEAD)]
    for k, (name, desc) in BLURS.items():
        desc = desc.replace("前処理: ", "")
        col2.append((k, desc_of(name, desc), COLOR_CURRENT if name == st.blur_proc else COLOR_ITEM))
    param = {"gaussian": "k=%d" % st.gaussian_scale,
             "median": "k=%d" % st.median_scale,
             "bilateral": "d=%d" % st.bilateral_scale}.get(st.blur_proc, "")
    profile = "ON" if st.profile_face else "OFF"
    smooth = "OFF" if (st.pose and not st.pose.smoothing) else "ON"
    if ja:
        col2 += [("↑ / +", "ぼかしを強く", COLOR_ITEM),
                 ("↓ / -", "ぼかしを弱く", COLOR_ITEM),
                 ("ESC", "ぼかしをリセット", COLOR_ITEM),
                 ("", "（現在: %s  %s）" % (st.blur_proc, param), COLOR_CURRENT),
                 ("", "", COLOR_ITEM),
                 ("", "その他", COLOR_HEAD),
                 ("p", "横顔検出 %s（顔検出 F の実行中）" % profile, COLOR_ITEM),
                 ("k", "骨格の平滑化 %s（骨格推定 K の実行中）" % smooth, COLOR_ITEM),
                 ("n", "NDI 送信 ON/OFF（任意機能）", COLOR_ITEM),
                 ("?", "このヘルプを閉じる / 開く", COLOR_ITEM),
                 ("q", "終了", COLOR_ITEM)]
    else:
        col2 += [("Up / +", "stronger blur", COLOR_ITEM),
                 ("Down / -", "weaker blur", COLOR_ITEM),
                 ("ESC", "reset blur", COLOR_ITEM),
                 ("", "(now: %s  %s)" % (st.blur_proc, param), COLOR_CURRENT),
                 ("", "", COLOR_ITEM),
                 ("", "Others", COLOR_HEAD),
                 ("p", "profile face %s (during F)" % profile, COLOR_ITEM),
                 ("k", "pose smoothing %s (during K)" % smooth, COLOR_ITEM),
                 ("n", "NDI output ON/OFF (optional)", COLOR_ITEM),
                 ("?", "close / open this help", COLOR_ITEM),
                 ("q", "quit", COLOR_ITEM)]
    return col1, col2


def status_px(H):
    """画面上の FPS 表示などの文字サイズ（画像の高さに合わせる）"""
    return max(H / 30, 12)


def render_help_layer(W, H, st, drawer):
    """黒地にヘルプの文字を描いた画像を作る"""
    ja = drawer.japanese
    title = ("キー操作（ウィンドウを選択した状態で押す）" if ja
             else "Key operations (click the window first)")
    col1, col2 = help_columns(st, ja)
    rows = max(len(col1), len(col2)) + 2          # +2 はタイトルと空行の分

    def col_size(col, px):
        """(キー欄の幅, 列全体の幅) を返す。見出しはキー欄を使わない"""
        kw = max(drawer.width(k, px) for k, _, _ in col if k) + px
        w = max((kw if k else 0) + drawer.width(t, px) for k, t, _ in col if t)
        return kw, max(w, max(drawer.width(t, px) for k, t, _ in col if not k and t))

    # 上端は FPS 表示の行のために空けておき、残りの高さに収まる文字サイズを決める。
    # 横にはみ出す場合はさらに縮める
    top = status_px(H) * 1.8
    area_h = H - top - H * 0.03
    px = min(max(area_h / rows / 1.45, 10), 40)
    for _ in range(3):
        (kw1, w1), (kw2, w2) = col_size(col1, px), col_size(col2, px)
        gap = px * 2.5
        total_w = max(w1 + gap + w2, drawer.width(title, px * 1.1))
        if total_w <= W * 0.92:
            break
        px *= W * 0.92 / total_w
    line_h = px * 1.45

    layer = np.zeros((H, W, 3), np.uint8)
    x0 = (W - total_w) / 2
    y0 = top + (area_h - rows * line_h) / 2 + px
    drawer.put(layer, title, (x0, y0), COLOR_ITEM, px * 1.1, bold=True)
    for col, x, kw in ((col1, x0, kw1), (col2, x0 + w1 + gap, kw2)):
        y = y0 + line_h * 2
        for k, text, color in col:
            bold = (color == COLOR_HEAD)
            if k:
                drawer.put(layer, k, (x, y), color, px, bold)
                drawer.put(layer, text, (x + kw, y), color, px, bold)
            elif text:
                drawer.put(layer, text, (x, y), color, px, bold)
            y += line_h
    return layer


def draw_help(img, st, drawer, cache):
    """画像を暗くしてヘルプを重ねる（文字の画像は内容が変わったときだけ作り直す）"""
    H, W = img.shape[:2]
    key = (W, H, st.proc, st.blur_proc, st.gaussian_scale, st.median_scale,
           st.bilateral_scale, st.profile_face, bool(st.pose and st.pose.smoothing))
    if cache.get("key") != key:
        cache["key"] = key
        cache["layer"] = render_help_layer(W, H, st, drawer)
    dark = cv2.convertScaleAbs(img, alpha=HELP_DIM)
    return cv2.max(dark, cache["layer"])


# ----------------------------------------------------------------------
# キー入力（OS によって矢印キーのコードが異なるので吸収する）
# ----------------------------------------------------------------------
KEY_UP_CODES = {2490368}      # Windows
KEY_DOWN_CODES = {2621440}    # Windows
KEY_UP_LOW16 = {0xF700, 0xFF52}    # macOS, Linux
KEY_DOWN_LOW16 = {0xF701, 0xFF54}  # macOS, Linux


def read_key():
    """溜まっているキー入力のうち最新のものだけを返す（それ以外は捨てる）"""
    code = -1
    while True:
        k = cv2.waitKeyEx(1)
        if k == -1:
            break
        code = k
    if code == -1:
        return None
    low16 = code & 0xFFFF
    if code in KEY_UP_CODES or low16 in KEY_UP_LOW16:
        return "UP"
    if code in KEY_DOWN_CODES or low16 in KEY_DOWN_LOW16:
        return "DOWN"
    if 0xF700 <= low16 <= 0xF8FF or 0xFF00 <= low16 <= 0xFFFF:
        return None  # その他の特殊キーは無視
    return chr(code & 0xFF)


# ----------------------------------------------------------------------
# NDI 送信（任意機能。ndi-python がインストールされている場合のみ）
# ----------------------------------------------------------------------
ndi = None


def NDI_setup():
    global ndi
    try:
        import NDIlib
        ndi = NDIlib
    except ModuleNotFoundError as e:
        print(e)
        print("NDI module not found, so this function is unavailable.")
        return None, None
    if not ndi.initialize():
        print("NDI module found but cannot be initialized.")
        return None, None
    print("NDI successfully initialized.")
    send_settings = ndi.SendCreate()
    send_settings.ndi_name = "ndi-python"
    ndi_send = ndi.send_create(send_settings)
    ndi_frame = ndi.VideoFrameV2()
    return ndi_send, ndi_frame


def NDI_finish(ndi_send):
    if ndi_send is not None:
        ndi.send_destroy(ndi_send)
        ndi.destroy()


# ----------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------
def open_source(arg):
    """引数が数字ならカメラ番号、そうでなければ動画ファイルとして開く"""
    if arg is None:
        arg = "0"
    if arg.isdigit():
        print("camera device num: %s" % arg)
        cap = cv2.VideoCapture(int(arg))
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap, False
    print("video file: %s" % arg)
    return cv2.VideoCapture(arg), True


def main():
    cap, is_file = open_source(sys.argv[1] if len(sys.argv) > 1 else None)

    ret, img = cap.read()
    if not ret:
        print("カメラ（または動画）から画像を取得できませんでした。")
        print("  - カメラ番号を変えて試してください（例: python capture.py 1）")
        print("  - macOS ではターミナルにカメラへのアクセス許可が必要です")
        print("    （システム設定 > プライバシーとセキュリティ > カメラ）")
        sys.exit(1)

    print("FPS:%f" % cap.get(cv2.CAP_PROP_FPS))
    print("Image Height:%d" % int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("Image Width:%d" % int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    height, width = img.shape[:2]
    proc_size = (int(width * PROC_SCALE), int(height * PROC_SCALE))
    st = State((proc_size[1], proc_size[0]))

    ndi_send, ndi_frame = None, None
    print_help()
    show_help = SHOW_HELP_AT_START
    drawer = TextDrawer()
    help_cache = {}

    t_prev = time.perf_counter()
    while True:
        t_now = time.perf_counter()
        duration = max(t_now - t_prev, 1e-6)
        t_prev = t_now

        # 画像の取り込み
        ret, img = cap.read()
        if not ret:
            if is_file:      # 動画ファイルは先頭に戻して繰り返す
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("image not captured. stop.")
            break

        # 処理速度のため縮小
        img = cv2.resize(img, proc_size)

        # キー入力
        key = read_key()
        if key == 'q':
            break
        elif key == "UP" or key == '+':
            st.increase_param()
        elif key == "DOWN" or key == '-':
            st.decrease_param()
        elif key == chr(27):          # ESC
            st.reset_param()
        elif key == '?':
            show_help = not show_help
            if show_help:
                print_help()
        elif key in BLURS:
            st.blur_proc = BLURS[key][0]
            st.reset_param()
            print("blur:", st.blur_proc)
        elif key in PROCS:
            st.proc = PROCS[key][0]
            if st.proc == "lktracking":
                st.lk_reset = True
            if st.proc == "pose" and st.pose:
                st.pose.reset()
            if st.proc in ("dog", "hist_eq", "hist_eq_CLAHE"):
                st.blur_proc = "gaussian"
            print("proc:", st.proc)
        elif key == 'k' and st.proc == "pose" and st.pose:
            st.pose.smoothing = not st.pose.smoothing
            st.pose.reset()
            print("pose smoothing:", "ON" if st.pose.smoothing else "OFF")
        elif key == 'p' and st.proc == "facedetect":
            st.profile_face = not st.profile_face
            print("profile face mode" if st.profile_face else "frontal face mode")
        elif key == 'n':
            if ndi_send is None:
                ndi_send, ndi_frame = NDI_setup()
            else:
                NDI_finish(ndi_send)
                ndi_send = None

        # 前処理（ぼかし）→ 画像処理
        img = blur(img, st)
        result = PROC_FUNCS[st.proc](img, st)

        # 表示
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        result = cv2.resize(result, (width, height))
        if show_help:
            result = draw_help(result, st, drawer, help_cache)
        px = status_px(height)
        drawer.put(result, "FPS:%.1f  proc:%s  blur:%s" % (1.0 / duration, st.proc, st.blur_proc),
                   (px * 0.5, px * 1.3), (0, 255, 0), px, bold=True)
        if not show_help:
            drawer.put(result, "?: ヘルプ" if drawer.japanese else "?: help",
                       (px * 0.5, height - px * 0.6), (0, 255, 0), px * 0.8)
        cv2.imshow(WINDOW_NAME, result)

        # NDI へ送信
        if ndi_send is not None:
            ndi_frame.data = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
            ndi_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
            ndi.send_send_video_v2(ndi_send, ndi_frame)

    NDI_finish(ndi_send)
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
