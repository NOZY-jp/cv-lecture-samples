# coding: utf-8
"""
MediaPipe による顔・表情・骨格・手の推定（mediapipe が必要）

    python mediapipe_demo.py            # カメラ 0
    python mediapipe_demo.py 1          # カメラ 1
    python mediapipe_demo.py movie.mp4  # 動画ファイル

キー操作（画像ウィンドウを選択した状態で。それぞれ独立に ON/OFF でき、同時に使える）:
    f : Face Mesh（顔の 478 点）と顔の向き
    b : 表情（Blendshapes）の強さのグラフ（Face Mesh が ON のとき）
    p : 骨格（Pose, 33 点）
    h : 手（Hand, 片手 21 点 x 2）
    m : 左右反転（鏡のように表示）
    q / ESC : 終了

capture.py の K キー（pose_estimation.py）は、MediaPipe の骨格推定のモデルを OpenCV で
動かして中身を見せるためのものです。こちらは Google の MediaPipe ライブラリそのものを使います。
"""

import contextlib
import os
import sys
import time

os.environ.setdefault("GLOG_minloglevel", "2")       # MediaPipe の詳細なログを抑える
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
except ImportError:
    print("mediapipe がインストールされていません。")
    print("  pip install -r requirements-mediapipe.txt  でインストールしてください。")
    print("  （Intel 製 CPU の Mac には対応していません）")
    sys.exit(1)

import pose_estimation      # 骨格の描き方を K キー版とそろえるため

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FACE_MODEL = os.path.join(MODEL_DIR, "face_landmarker.task")
POSE_MODEL = os.path.join(MODEL_DIR, "pose_landmarker_full.task")
HAND_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")


# ----------------------------------------------------------------------
# 推定器の作成（VIDEO モード: 前のフレームの結果を使って追跡・平滑化してくれる）
# ----------------------------------------------------------------------
@contextlib.contextmanager
def quiet_stderr():
    """MediaPipe の内部（C++）がモデルの読み込み時に出す大量の警告を一時的に隠す。
    （読み込みに失敗した場合は Python の例外として表示される）"""
    try:
        fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    sys.stderr.flush()
    saved, null = os.dup(fd), os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, fd)
    try:
        yield
    finally:
        os.dup2(saved, fd)
        os.close(null)
        os.close(saved)


def create_landmarker(kind):
    path = {"face": FACE_MODEL, "pose": POSE_MODEL, "hand": HAND_MODEL}[kind]
    if not os.path.exists(path):
        print("モデルがありません: %s\npython download_models.py を実行してください。" % path)
        return None
    with quiet_stderr():
        return _create(kind, BaseOptions(model_asset_path=path), vision.RunningMode.VIDEO)


def _create(kind, base, video):
    if kind == "face":
        opts = vision.FaceLandmarkerOptions(
            base_options=base, running_mode=video, num_faces=1,
            output_face_blendshapes=True,                 # 表情の強さ
            output_facial_transformation_matrixes=True)   # 顔の向き（4x4 の変換行列）
        return vision.FaceLandmarker.create_from_options(opts)
    if kind == "pose":
        opts = vision.PoseLandmarkerOptions(base_options=base, running_mode=video, num_poses=1)
        return vision.PoseLandmarker.create_from_options(opts)
    opts = vision.HandLandmarkerOptions(base_options=base, running_mode=video, num_hands=2)
    return vision.HandLandmarker.create_from_options(opts)


def to_pixels(landmarks, w, h):
    """正規化座標（0〜1）のリスト → 画素座標の配列 (N, 2)"""
    return np.array([[p.x * w, p.y * h] for p in landmarks])


def segments(pts, connections):
    """線のつなぎ方のリスト → cv2.polylines でまとめて描ける線分の配列"""
    idx = np.array([[c.start, c.end] for c in connections])
    return np.round(pts[idx]).astype(np.int32)              # (線の本数, 2, 2)


# ----------------------------------------------------------------------
# 顔: Face Mesh と顔の向き
# ----------------------------------------------------------------------
FC = vision.FaceLandmarksConnections
FACE_TESSELATION = FC.FACE_LANDMARKS_TESSELATION
FACE_CONTOURS = FC.FACE_LANDMARKS_CONTOURS
FACE_IRISES = FC.FACE_LANDMARKS_LEFT_IRIS + FC.FACE_LANDMARKS_RIGHT_IRIS
NOSE_TIP = 1


def head_angles(matrix):
    """顔の変換行列（4x4）から顔の向きを求める。

    変換行列の左上 3x3 は、正面を向いた標準の顔をカメラ座標（x: 右, y: 上, z: 手前）に
    置くための回転 R。これを R = Rz(roll)・Ry(yaw)・Rx(pitch) と分解すると、
    首をかしげても（roll）左右（yaw）・上下（pitch）の角度が変わらない。
    yaw: 画面の右を向くと正, pitch: 上を向くと正, roll: 画面上で反時計回りに傾けると正 [度]
    """
    R = np.array(matrix)[:3, :3]
    R = R / np.linalg.norm(R, axis=0)                        # 拡大縮小の成分を除く
    yaw = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
    pitch = -np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll, R


def draw_face(img, result, show_blendshapes):
    h, w = img.shape[:2]
    if not result.face_landmarks:
        return
    pts = to_pixels(result.face_landmarks[0], w, h)
    cv2.polylines(img, segments(pts, FACE_TESSELATION), False, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.polylines(img, segments(pts, FACE_CONTOURS), False, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.polylines(img, segments(pts, FACE_IRISES), False, (0, 255, 255), 1, cv2.LINE_AA)

    # 顔の向き: 鼻先から、顔の右（赤）・上（緑）・正面（青）の方向に線を引く
    if result.facial_transformation_matrixes:
        yaw, pitch, roll, R = head_angles(result.facial_transformation_matrixes[0])
        size = (pts[:, 0].max() - pts[:, 0].min()) * 0.6
        o = pts[NOSE_TIP]
        for axis, color in ((0, (0, 0, 255)), (1, (0, 255, 0)), (2, (255, 0, 0))):
            v = R[:, axis]
            end = o + size * np.array([v[0], -v[1]])         # 画面の y は下向きなので反転
            cv2.arrowedLine(img, tuple(np.round(o).astype(int)), tuple(np.round(end).astype(int)),
                            color, 2, cv2.LINE_AA, tipLength=0.15)
        x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].max()) + 25
        cv2.putText(img, "yaw %+5.1f  pitch %+5.1f  roll %+5.1f" % (yaw, pitch, roll),
                    (x0, min(y0, h - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    if show_blendshapes and result.face_blendshapes:
        draw_blendshapes(img, result.face_blendshapes[0])


def draw_blendshapes(img, blendshapes, top=12):
    """表情の強さ（0〜1）の大きいものから top 個を棒グラフで表示"""
    h, w = img.shape[:2]
    items = sorted((b for b in blendshapes if b.category_name != "_neutral"),
                   key=lambda b: b.score, reverse=True)[:top]
    x0, y0, row, bar = w - 330, 70, 24, 140
    panel = img[y0 - 22:y0 + row * len(items), x0 - 10:w - 10]
    panel[:] = (panel * 0.4).astype(np.uint8)                # 背景を暗くして読みやすく
    for i, b in enumerate(items):
        y = y0 + i * row
        cv2.putText(img, b.category_name, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0 + 160, y - 12), (x0 + 160 + int(bar * b.score), y - 2),
                      (0, 200, 255), -1)


# ----------------------------------------------------------------------
# 骨格と手
# ----------------------------------------------------------------------
def line_thickness(img):
    """骨格・手の線の太さ（画像の大きさに合わせる。720p で 3、1080p で 4〜5 画素）"""
    return max(2, round(min(img.shape[:2]) / 240))


def draw_pose(img, result):
    h, w = img.shape[:2]
    for landmarks in result.pose_landmarks:
        # K キー版（pose_estimation.py）と同じ形式 (x, y, z, visibility, presence) にして描く
        lm = np.array([[p.x * w, p.y * h, p.z, p.visibility or 0.0, p.presence or 0.0]
                       for p in landmarks])
        pose_estimation.draw_pose(img, lm, thickness=line_thickness(img))


HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
HAND_COLOR = {"Left": (0, 160, 255), "Right": (255, 200, 0)}


def draw_hands(img, result):
    h, w = img.shape[:2]
    for landmarks, handed in zip(result.hand_landmarks, result.handedness):
        label = handed[0].category_name                       # "Left" / "Right"
        color = HAND_COLOR.get(label, (255, 255, 255))
        pts = to_pixels(landmarks, w, h)
        t = line_thickness(img)
        cv2.polylines(img, segments(pts, HAND_CONNECTIONS), False, color, t, cv2.LINE_AA)
        for p in pts:
            # 手は関節が密なので少し小さめに。円周は線の色と区別できるよう黒の 1 重にする
            pose_estimation.draw_joint(img, p, max(t - 1, 2), (0, 0, 0), outline=False)
        wrist = np.round(pts[0]).astype(int)
        cv2.putText(img, label, (int(wrist[0]) - 20, int(wrist[1]) + 15 + t * 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


# ----------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------
HELP = """キー操作（画像ウィンドウを選択した状態で）:
  f : Face Mesh（顔の 478 点）と顔の向き ON/OFF
  b : 表情（Blendshapes）のグラフ ON/OFF（Face Mesh が ON のとき）
  p : 骨格（Pose）ON/OFF
  h : 手（Hand）ON/OFF
  m : 左右反転 ON/OFF
  q / ESC : 終了"""


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    cap = cv2.VideoCapture(int(arg) if arg.isdigit() else arg)
    if not cap.isOpened():
        print("カメラ（または動画）を開けませんでした: %s" % arg)
        sys.exit(1)
    print(HELP)

    enabled = {"face": True, "pose": False, "hand": False}
    landmarkers = {}            # 必要になったときに作る
    warmed = set()
    show_blendshapes, mirror = True, True
    t_prev, ts_prev = time.perf_counter(), -1

    while True:
        ret, img = cap.read()
        if not ret:
            if not arg.isdigit():                    # 動画は先頭に戻して繰り返す
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        if mirror:
            img = cv2.flip(img, 1)

        # VIDEO モードでは、フレームごとに増えていく時刻 [ミリ秒] を渡す
        ts = max(int(time.perf_counter() * 1000), ts_prev + 1)
        ts_prev = ts
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        for kind in ("pose", "hand", "face"):        # 顔を最後に描く（いちばん上に表示）
            if not enabled[kind]:
                continue
            if kind not in landmarkers:
                landmarkers[kind] = create_landmarker(kind)
            if landmarkers[kind] is None:            # モデルがない
                enabled[kind] = False
                continue
            # 1 回目の推定では内部の注意書きが出るので隠す
            with quiet_stderr() if kind not in warmed else contextlib.nullcontext():
                result = landmarkers[kind].detect_for_video(frame, ts)
            warmed.add(kind)
            if kind == "face":
                draw_face(img, result, show_blendshapes)
            elif kind == "pose":
                draw_pose(img, result)
            else:
                draw_hands(img, result)

        t_now = time.perf_counter()
        fps, t_prev = 1.0 / max(t_now - t_prev, 1e-6), t_now
        onoff = lambda b: "ON" if b else "OFF"
        status = "FPS:%.1f  [f]Face:%s [b]Blend:%s [p]Pose:%s [h]Hand:%s [m]Mirror:%s  [q]quit" % (
            fps, onoff(enabled["face"]), onoff(show_blendshapes), onoff(enabled["pose"]),
            onoff(enabled["hand"]), onoff(mirror))
        # ふちは同じ太さの黒い文字をずらして重ねる（OpenCV 5 では太さで文字の幅が変わるため）
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, -1), (-1, 1), (1, 1)):
            cv2.putText(img, status, (10 + dx, 30 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("MediaPipe demo", img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('f'):
            enabled["face"] = not enabled["face"]
        elif key == ord('p'):
            enabled["pose"] = not enabled["pose"]
        elif key == ord('h'):
            enabled["hand"] = not enabled["hand"]
        elif key == ord('b'):
            show_blendshapes = not show_blendshapes
        elif key == ord('m'):
            mirror = not mirror

    with quiet_stderr():
        for lm in landmarkers.values():
            if lm is not None:
                lm.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
