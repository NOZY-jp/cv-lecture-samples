# coding: utf-8
"""
人体の骨格推定（MediaPipe BlazePose のモデルを OpenCV の DNN モジュールで動かす）

capture.py の K キーから使われます。このファイル単体でも動かせます。

    python pose_estimation.py        # カメラ 0 で骨格推定（s キーで平滑化 ON/OFF、q で終了）
    python pose_estimation.py 1      # カメラ 1

処理の流れ（PoseEstimator.process）:

    1. 切り出し位置を決める
         ・最初のフレーム（または見失ったとき）は「人物検出」で人を探す
         ・それ以降は、前のフレームの骨格推定の結果から決める（追跡）
    2. 骨格推定: 人物を囲む正方形を、体がまっすぐ立つ向きに回転させて切り出し、
       33 個の関節点を推定する。モデルは関節の座標と、関節ごとの「ありそうな場所」を
       表す画像（ヒートマップ）の両方を出力するので、ヒートマップで座標を細かく補正する
    3. 平滑化: 推定結果は毎フレーム少しずつ揺れるので、時間方向になめらかにする
       （One Euro Filter）
    4. 平滑化した結果から、次のフレームの切り出し位置を決める（→ 1 へ）

これは MediaPipe Pose（ライブラリ）が内部で行っている処理と同じ構成です。
モデルは OpenCV Zoo (https://github.com/opencv/opencv_zoo) で公開されているもので、
前処理・後処理は OpenCV Zoo の mp_persondet.py / mp_pose.py（Apache License 2.0）を、
ヒートマップによる補正と平滑化は MediaPipe（Apache License 2.0）の
refine_landmarks_from_heatmap_calculator.cc と pose_landmark_filtering.pbtxt を参考にしました。
"""

import os
import sys
import time

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PERSON_DET_MODEL = os.path.join(MODEL_DIR, "person_detection_mediapipe_2023mar.onnx")
POSE_MODEL = os.path.join(MODEL_DIR, "pose_estimation_mediapipe_2023mar.onnx")

# 推定される 39 点のうち、0〜32 が関節点、33 と 34 が切り出し位置を決めるための補助点
#   33: 腰の中心、34: 全身の大きさと向きを表す点（頭の上のほう）
NUM_JOINTS = 33
AUX_HIP, AUX_FULL = 33, 34


# ----------------------------------------------------------------------
# One Euro Filter（時間方向の平滑化）
#   Casiez et al., "1€ Filter", CHI 2012
#   ゆっくり動くときは強くならして揺れを消し、速く動くときは弱くならして遅れを防ぐ。
# ----------------------------------------------------------------------
class OneEuroFilter:
    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff = min_cutoff    # 止まっているときのカットオフ周波数 [Hz]（小さいほど強くならす）
        self.beta = beta                # 速さに応じてカットオフ周波数を上げる度合い
        self.d_cutoff = d_cutoff        # 速さの推定に使うカットオフ周波数 [Hz]
        self.reset()

    def reset(self):
        self.x_raw = None       # 前回の入力
        self.x_hat = None       # 前回の出力
        self.dx_hat = None      # 前回の速さ（平滑化済み）
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        """カットオフ周波数から指数移動平均の係数を求める"""
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t, scale=1.0):
        """x: 入力（配列可、要素ごとに独立に平滑化）, t: 時刻 [秒],
        scale: 速さをこの値で割る（人物の大きさで割れば、遠い人も近い人も同じ扱いになる）"""
        x = np.asarray(x, np.float64)
        if self.x_hat is None or t <= self.t:
            self.x_raw, self.x_hat, self.dx_hat, self.t = x, x, np.zeros_like(x), t
            return x
        dt = t - self.t
        # 速さ（人物の大きさ毎秒）を求めて、それ自体も少しならす
        dx = (x - self.x_raw) / scale / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx_hat = a_d * dx + (1 - a_d) * self.dx_hat
        # 速いほどカットオフ周波数を上げる（＝ならし方を弱める）
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx_hat)
        a = self._alpha(cutoff, dt)
        self.x_hat = a * x + (1 - a) * self.x_hat
        self.x_raw, self.t = x, t
        return self.x_hat


# ----------------------------------------------------------------------
# ヒートマップによる座標の補正
# ----------------------------------------------------------------------
def refine_with_heatmap(xy, heatmap, input_size, kernel_size=7, min_confidence=0.5):
    """モデルが直接出力した座標（xy, 入力画像の画素単位）を、ヒートマップで補正する。

    座標の直接出力（回帰）は、平均的な姿勢に引き寄せられて細かい位置がずれやすい。
    そこで、各関節のヒートマップ（64x64, 値が大きいほどその関節がありそうな場所）の
    推定位置のまわり kernel_size x kernel_size の範囲で、値を重みとした重心を求め直す。
    範囲内の最大値が min_confidence 未満なら、ヒートマップが当てにならないので補正しない。
    """
    hm_h, hm_w, n = heatmap.shape
    prob = 1 / (1 + np.exp(-np.clip(heatmap, -50, 50)))       # ロジット → 確率
    half = (kernel_size - 1) // 2
    xy = xy.copy()
    for i in range(n):
        col = int(xy[i, 0] / input_size * hm_w)                # ヒートマップ上の位置
        row = int(xy[i, 1] / input_size * hm_h)
        if not (0 <= col < hm_w and 0 <= row < hm_h):
            continue
        c0, c1 = max(0, col - half), min(hm_w, col + half + 1)
        r0, r1 = max(0, row - half), min(hm_h, row + half + 1)
        w = prob[r0:r1, c0:c1, i]
        if w.max() < min_confidence or w.sum() <= 0:
            continue
        rows, cols = np.mgrid[r0:r1, c0:c1]
        xy[i, 0] = (cols * w).sum() / w.sum() / hm_w * input_size   # 重み付き重心
        xy[i, 1] = (rows * w).sum() / w.sum() / hm_h * input_size
    return xy


# ----------------------------------------------------------------------
# 人物検出器のアンカー
# ----------------------------------------------------------------------
def make_ssd_anchors(input_size=224, strides=(8, 16, 32, 32, 32), offset=0.5):
    """人物検出器のアンカー（候補枠の基準位置）を作る。
    特徴マップの格子点ごとに候補を置く SSD 方式で、同じストライドの層はまとめ、
    1 層につき 2 個ずつ候補を置く（224 画素入力なら 28x28x2 + 14x14x2 + 7x7x6 = 2254 個）"""
    anchors = []
    i = 0
    while i < len(strides):
        stride, n = strides[i], 0
        while i < len(strides) and strides[i] == stride:
            n += 2
            i += 1
        fm = int(np.ceil(input_size / stride))
        for y in range(fm):
            for x in range(fm):
                anchors += [[(x + offset) / fm, (y + offset) / fm]] * n
    return np.array(anchors, np.float32)


# ----------------------------------------------------------------------
# 骨格推定器
# ----------------------------------------------------------------------
class PoseEstimator:
    DET_SIZE = 224      # 人物検出器の入力サイズ
    POSE_SIZE = 256     # 骨格推定器の入力サイズ
    ROI_SCALE = 1.25    # 2 点から決めた正方形をこの倍率に広げて切り出す（MediaPipe と同じ）

    def __init__(self, det_model=PERSON_DET_MODEL, pose_model=POSE_MODEL,
                 det_score=0.5, pose_conf=0.5, smoothing=True):
        self.det = cv2.dnn.readNet(det_model)
        self.pose = cv2.dnn.readNet(pose_model)
        self.anchors = make_ssd_anchors(self.DET_SIZE)
        self.det_score = det_score      # 人物検出のスコアの閾値
        self.pose_conf = pose_conf      # 骨格推定の信頼度の閾値（下回ったら見失ったとみなす）
        self.smoothing = smoothing      # 時間方向の平滑化を行うか
        # 平滑化のパラメータは MediaPipe と同じ
        self.joint_filter = OneEuroFilter(min_cutoff=0.05, beta=80.0)   # 関節点
        self.aux_filter = OneEuroFilter(min_cutoff=0.01, beta=10.0)     # 補助点（切り出し位置用）
        self.visibility = None
        self.roi = None     # 次のフレームの切り出し位置 (腰の中心, 全身点)。None なら人物検出から

    def reset(self):
        self.roi = None
        self.visibility = None
        self.joint_filter.reset()
        self.aux_filter.reset()

    # --- 1. 人物検出 ---
    def detect_person(self, img):
        """最もスコアの高い 1 人を探し、(腰の中心, 全身点) を返す。いなければ None"""
        h, w = img.shape[:2]
        s = self.DET_SIZE
        ratio = s / max(h, w)
        nw, nh = int(w * ratio), int(h * ratio)
        rgb = cv2.cvtColor(cv2.resize(img, (nw, nh)), cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = rgb / 127.5 - 1.0                                  # 画素値を [-1, 1] に
        left, top = (s - nw) // 2, (s - nh) // 2                 # 縦横比を保って正方形に
        pad = cv2.copyMakeBorder(rgb, top, s - nh - top, left, s - nw - left,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
        self.det.setInput(np.ascontiguousarray(pad.transpose(2, 0, 1)[np.newaxis]))  # NCHW
        outs = self.det.forward(self.det.getUnconnectedOutLayersNames())
        reg = [o for o in outs if o.shape[-1] == 12][0][0]       # 候補ごとの枠と 4 点の回帰値
        score = [o for o in outs if o.shape[-1] == 1][0][0, :, 0]
        score = 1 / (1 + np.exp(-np.clip(score.astype(np.float64), -100, 100)))
        best = int(np.argmax(score))
        if score[best] < self.det_score:
            return None
        # 4 点: 腰の中心, 全身点, 肩の中心, 上半身点（アンカーからのずれ）
        kp = reg[best, 4:].reshape(4, 2) / s + self.anchors[best]
        kp = (kp * s - np.array([left, top])) / ratio            # 元の画像の座標へ
        return kp[0], kp[1]

    # --- 2. 骨格推定 ---
    def estimate(self, img, hip, full):
        """(腰の中心, 全身点) から切り出して骨格を推定する。
        (39x5 の点, 信頼度) を返す。各点は x, y, z, visibility, presence。見失ったら None"""
        # 腰の中心〜全身点の距離の 2 倍を一辺とする正方形を ROI_SCALE 倍に広げる
        r = np.linalg.norm(full - hip) * self.ROI_SCALE
        if r < 4:
            return None
        # 全身点が腰の真上に来るように回転し、腰を中心とする 2r x 2r の正方形を 256x256 に切り出す
        angle = np.degrees(np.arctan2(full[0] - hip[0], -(full[1] - hip[1])))
        s = self.POSE_SIZE
        M = cv2.getRotationMatrix2D((float(hip[0]), float(hip[1])), angle, s / (2 * r))
        M[:, 2] += np.array([s / 2, s / 2]) - hip
        crop = cv2.warpAffine(img, M, (s, s), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        blob = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)[np.newaxis] / 255.0  # NHWC
        self.pose.setInput(blob)
        outs = self.pose.forward(self.pose.getUnconnectedOutLayersNames())
        by_size = {o.size: o for o in outs}
        conf = float(by_size[1].ravel()[0])
        if conf < self.pose_conf:
            return None
        lm = by_size[195].reshape(39, 5).astype(np.float64)
        lm[:, 3:] = 1 / (1 + np.exp(-lm[:, 3:]))                 # visibility, presence を確率に
        # ヒートマップ（64x64x39）で座標を補正
        heatmap = by_size[64 * 64 * 39].reshape(64, 64, 39)
        lm[:, :2] = refine_with_heatmap(lm[:, :2], heatmap, s)
        # 切り出し画像の座標 → 元の画像の座標（アフィン変換の逆変換）
        Minv = cv2.invertAffineTransform(M)
        lm[:, :2] = np.c_[lm[:, :2], np.ones(len(lm))] @ Minv.T
        return lm, conf

    # --- 3. 平滑化 ---
    def smooth(self, lm, t):
        """関節点と補助点を時間方向になめらかにする"""
        lm = lm.copy()
        # 速さを「人物の大きさ」で割るための値（補助点 2 点の距離の 2 倍）
        scale = max(2 * np.linalg.norm(lm[AUX_FULL, :2] - lm[AUX_HIP, :2]), 1e-6)
        lm[:NUM_JOINTS, :3] = self.joint_filter(lm[:NUM_JOINTS, :3], t, scale)
        lm[NUM_JOINTS:, :3] = self.aux_filter(lm[NUM_JOINTS:, :3], t, scale)
        # 見えている確率（visibility）は単純な指数移動平均でならす
        v = lm[:, 3]
        self.visibility = v if self.visibility is None else 0.1 * v + 0.9 * self.visibility
        lm[:, 3] = self.visibility
        return lm

    # --- 1〜4 をまとめた処理 ---
    def process(self, img, t=None):
        """1 フレーム分の処理。(39x5 の点, 信頼度) を返す。見つからなければ None"""
        t = time.perf_counter() if t is None else t

        # 1. 切り出し位置: 追跡中なら前のフレームの結果、そうでなければ人物検出
        res = None
        if self.roi is not None:
            res = self.estimate(img, *self.roi)
        if res is None:                       # 見失った → 平滑化をリセットして人物検出から
            self.reset()
            person = self.detect_person(img)
            if person is None:
                return None
            # 2. 骨格推定
            res = self.estimate(img, *person)
            if res is None:
                return None
        lm, conf = res

        # 3. 平滑化
        if self.smoothing:
            lm = self.smooth(lm, t)

        # 4. 次のフレームの切り出し位置（平滑化した補助点から決める）
        self.roi = (lm[AUX_HIP, :2].copy(), lm[AUX_FULL, :2].copy())
        return lm, conf


# ----------------------------------------------------------------------
# 描画
# ----------------------------------------------------------------------
# 骨格の線のつなぎ方（33 点）。l=左半身（オレンジ）, r=右半身（水色）, c=中央（白）
POSE_EDGES = [
    (0, 1, 'c'), (1, 2, 'l'), (2, 3, 'l'), (3, 7, 'l'), (0, 4, 'c'), (4, 5, 'r'), (5, 6, 'r'),
    (6, 8, 'r'), (9, 10, 'c'), (11, 12, 'c'), (11, 23, 'l'), (12, 24, 'r'), (23, 24, 'c'),
    (11, 13, 'l'), (13, 15, 'l'), (15, 17, 'l'), (15, 19, 'l'), (15, 21, 'l'), (17, 19, 'l'),
    (12, 14, 'r'), (14, 16, 'r'), (16, 18, 'r'), (16, 20, 'r'), (16, 22, 'r'), (18, 20, 'r'),
    (23, 25, 'l'), (25, 27, 'l'), (27, 29, 'l'), (27, 31, 'l'), (29, 31, 'l'),
    (24, 26, 'r'), (26, 28, 'r'), (28, 30, 'r'), (28, 32, 'r'), (30, 32, 'r'),
]
SIDE_COLOR = {'l': (0, 160, 255), 'r': (255, 200, 0), 'c': (255, 255, 255)}


def draw_joint(img, p, thickness=2, color=(255, 255, 255), ring=3, outline=True):
    """関節を描く: 中心の赤い点と、それを囲む大きな円の円周（半径は thickness の ring 倍）。
    円周は color で、outline=True ならどんな背景でも見えるように黒いふちを付ける"""
    p = (int(round(p[0])), int(round(p[1])))
    r, w = thickness * ring, max(thickness // 2, 1)
    if outline:
        cv2.circle(img, p, r, (0, 0, 0), w + 2, cv2.LINE_AA)    # 円周の黒いふち
    cv2.circle(img, p, r, color, w, cv2.LINE_AA)                # 円周
    cv2.circle(img, p, thickness, (0, 0, 255), -1, cv2.LINE_AA)  # 中心の点


# 関節ごとの円周の色（線と同じく左半身=オレンジ、右半身=水色、鼻=白）
JOINT_SIDE = ['c'] + ['l', 'l', 'l', 'r', 'r', 'r', 'l', 'r', 'l', 'r'] + ['l', 'r'] * 11


def draw_pose(img, lm, conf=None, thickness=2):
    """画像に骨格を描く（画面内にあると推定された点だけ）"""
    pts = np.round(lm[:NUM_JOINTS, :2]).astype(int)
    ok = lm[:NUM_JOINTS, 4] > 0.5
    for a, b, side in POSE_EDGES:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), SIDE_COLOR[side], thickness, cv2.LINE_AA)
    for i in np.flatnonzero(ok):
        # 顔の点（0〜10）は密集しているので円を小さめにする
        draw_joint(img, pts[i], thickness, SIDE_COLOR[JOINT_SIDE[i]], ring=2 if i <= 10 else 3)
    if conf is not None and ok.any():         # 信頼度を骨格の左上に表示
        x, y = pts[ok][:, 0].min(), pts[ok][:, 1].min()
        cv2.putText(img, "%.2f" % conf, (int(x), max(int(y) - 10 - thickness * 3, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------
# 単体で動かす場合
# ----------------------------------------------------------------------
def main():
    for p in (PERSON_DET_MODEL, POSE_MODEL):
        if not os.path.exists(p):
            print("モデルがありません: %s\npython download_models.py を実行してください。" % p)
            sys.exit(1)
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(cam)
    estimator = PoseEstimator()
    print("s: 平滑化 ON/OFF   q: 終了")
    while True:
        ret, img = cap.read()
        if not ret:
            break
        res = estimator.process(img)
        if res is not None:
            draw_pose(img, *res, thickness=max(2, round(min(img.shape[:2]) / 240)))
        cv2.putText(img, "smoothing: %s (s)" % ("ON" if estimator.smoothing else "OFF"), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("pose estimation", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('s'):
            estimator.smoothing = not estimator.smoothing
            estimator.reset()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
