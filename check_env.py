# coding: utf-8
"""
実行環境のチェック用スクリプト

    python check_env.py          # カメラ 0 をチェック
    python check_env.py 1        # カメラ 1 をチェック

必要なモジュールがインストールされているか、各アルゴリズムが使えるか、
カメラから画像が取れるかを確認します。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REINSTALL = ("pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless "
             "opencv-contrib-python-headless のあと pip install -r requirements.txt")
ok = True


def result(flag, msg, hint=None, required=True):
    """required=False の項目は NG でも全体の判定には影響しない（参考情報）"""
    global ok
    mark = "OK" if flag else ("NG" if required else "--")
    print("[%s] %s" % (mark, msg))
    if not flag:
        if required:
            ok = False
        if hint:
            print("      -> " + hint)


print("Python %s (%s)" % (sys.version.split()[0], sys.executable))
result(sys.version_info >= (3, 8), "Python 3.8 以上", "新しい Python をインストールしてください")

# 仮想環境 (.venv) の中で実行されているか
in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
venv_dir = os.path.join(HERE, ".venv")
in_this_venv = in_venv and os.path.realpath(sys.prefix) == os.path.realpath(venv_dir)
if in_this_venv:
    result(True, "このフォルダの仮想環境 (.venv) で実行中")
else:
    if in_venv:
        where = "別の仮想環境 (%s)" % sys.prefix
    elif os.environ.get("CONDA_DEFAULT_ENV"):
        where = "Anaconda の環境「%s」" % os.environ["CONDA_DEFAULT_ENV"]
    else:
        where = "パソコン全体の Python"
    activate = (r".venv\Scripts\activate" if os.name == "nt" else "source .venv/bin/activate")
    if os.path.isdir(venv_dir):
        hint = "%s で仮想環境を有効にしてから実行してください" % activate
    else:
        hint = ("setup.bat" if os.name == "nt" else "sh setup.sh") + " でセットアップしてください"
    result(False, "仮想環境 (.venv) ではなく %s で実行しています（このまま動けば問題ありません）" % where,
           hint, required=False)

try:
    import numpy as np
    result(True, "numpy %s" % np.__version__)
except ImportError:
    result(False, "numpy が見つかりません", "pip install -r requirements.txt")

# インストールされている OpenCV のパッケージ（複数入っていると衝突する）
try:
    from importlib import metadata
    dists = []
    for name in ["opencv-python", "opencv-contrib-python",
                 "opencv-python-headless", "opencv-contrib-python-headless"]:
        try:
            dists.append("%s %s" % (name, metadata.version(name)))
        except metadata.PackageNotFoundError:
            pass
    if dists:
        result(len(dists) == 1, "OpenCV パッケージ: " + ", ".join(dists),
               "OpenCV のパッケージが複数入っています: " + REINSTALL)
        result(any(d.startswith("opencv-contrib-python ") for d in dists),
               "opencv-contrib-python（旧来のアルゴリズムも使える版）",
               REINSTALL)
except ImportError:
    pass

try:
    import cv2
    result(True, "opencv %s" % cv2.__version__)
except ImportError:
    result(False, "opencv (cv2) が見つかりません", "pip install -r requirements.txt")
    print("\nNG があります。上の指示に従ってください。")
    sys.exit(1)

# 画面表示（headless 版では imshow が使えない）
info = cv2.getBuildInformation()
gui_line = [l for l in info.splitlines() if l.strip().startswith("GUI:")]
gui = gui_line[0].split(":", 1)[1].strip() if gui_line else "UNKNOWN"
result(gui not in ("NONE", ""), "GUI バックエンド: %s" % gui,
       "headless 版が入っています: " + REINSTALL)


def find_cv_function(name):
    for mod in (cv2, getattr(cv2, "xfeatures2d", None)):
        if mod is not None and hasattr(mod, name):
            return getattr(mod, name)
    return None


def can_create(name, **kwargs):
    f = find_cv_function(name)
    if f is None:
        return False
    try:
        f(**kwargs)
        return True
    except cv2.error:
        return False


print()
print("--- 局所特徴量 (flann.py --feature) ---")
result(can_create("SIFT_create"), "SIFT")
result(can_create("ORB_create"), "ORB")
result(can_create("AKAZE_create"), "AKAZE", REINSTALL)
result(can_create("KAZE_create"), "KAZE", REINSTALL)
result(can_create("BRISK_create"), "BRISK", REINSTALL)
result(can_create("SURF_create"), "SURF（参考: 特許のため pip 版では使えないのが普通です）",
       required=False)

print("--- ロバスト推定 (flann.py --method) ---")
result(hasattr(cv2, "RANSAC"), "RANSAC")
result(hasattr(cv2, "USAC_MAGSAC"), "MAGSAC++", "pip install --upgrade -r requirements.txt")

print("--- 顔検出 (capture.py の F / Y キー) ---")
casc = os.path.join(HERE, "models", "haarcascades", "haarcascade_frontalface_default.xml")
result(hasattr(cv2, "CascadeClassifier"), "Haar-like 特徴カスケード (CascadeClassifier)", REINSTALL)
result(os.path.exists(casc), "カスケードファイル (models/haarcascades/)",
       "models フォルダごとダウンロードし直してください")
yunet = os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx")
result(hasattr(cv2, "FaceDetectorYN"), "YuNet (FaceDetectorYN)",
       "pip install --upgrade -r requirements.txt")
result(os.path.exists(yunet), "YuNet モデル (models/face_detection_yunet_2023mar.onnx)",
       "models フォルダごとダウンロードし直してください")

print("--- 人体の骨格推定 (capture.py の K キー) ---")
pose_files = [os.path.join(HERE, "models", f) for f in
              ("person_detection_mediapipe_2023mar.onnx", "pose_estimation_mediapipe_2023mar.onnx")]
result(all(os.path.exists(f) for f in pose_files), "MediaPipe Pose モデル (models/)",
       "python download_models.py でダウンロードしてください（骨格推定以外はこのままでも使えます）",
       required=False)

print("--- MediaPipe（任意: mediapipe_demo.py）---")
try:
    import importlib.util as _u
    has_mp = _u.find_spec("mediapipe") is not None
    mp_ver = metadata.version("mediapipe") if has_mp else ""
except Exception:
    has_mp, mp_ver = False, ""
result(has_mp, "mediapipe %s" % mp_ver if has_mp else "mediapipe",
       "pip install -r requirements-mediapipe.txt（Intel 製 CPU の Mac には非対応）", required=False)
if has_mp:
    tasks = [os.path.join(HERE, "models", f) for f in
             ("face_landmarker.task", "pose_landmarker_full.task", "hand_landmarker.task")]
    result(all(os.path.exists(f) for f in tasks), "MediaPipe のモデル (models/*.task)",
           "python download_models.py でダウンロードしてください", required=False)

print("--- 実行記録（課題の提出用）---")
try:
    sys.path.insert(0, HERE)
    import receipt
    inst = receipt.ensure_install_id()
    result(inst["event"] != "mismatch", "インストール ID: %s…（%s 作成）" % (inst["local_id"][:8], inst["local_created"]),
           "このフォルダは別の環境からコピーされた可能性があります。自分で clone し直してください",
           required=False)
except Exception as e:
    result(False, "インストール ID を作成できません (%s)" % e, required=False)

print("--- カメラ ---")
cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(cam)
ret, img = cap.read() if cap.isOpened() else (False, None)
if ret:
    result(True, "カメラ %d から画像を取得 (%dx%d)" % (cam, img.shape[1], img.shape[0]))
else:
    result(False, "カメラ %d から画像を取得できません" % cam,
           "カメラ番号を変える (python check_env.py 1) / 他のアプリがカメラを使っていないか確認 / "
           "macOS はターミナル等にカメラの許可を与える")
cap.release()

print()
print("すべて OK です。capture.py / flann.py を試してみましょう。" if ok
      else "NG があります。上の指示に従ってください。")
