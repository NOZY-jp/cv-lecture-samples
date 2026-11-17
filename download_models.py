# coding: utf-8
"""
学習済みモデルのダウンロード

    python download_models.py

次のモデルを models/ にダウンロードします。setup.sh / setup.bat から自動的に実行されます。
すでに正しいファイルがあればダウンロードしません。

  ・capture.py の K キー（骨格推定）用: OpenCV Zoo の MediaPipe Pose（約 18MB）
  ・mediapipe_demo.py 用: MediaPipe の Face / Pose / Hand Landmarker（約 21MB）
    （mediapipe がインストールされている場合のみ）
"""

import hashlib
import importlib.util
import os
import ssl
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")

# (ファイル名, [(ダウンロード元の URL, SHA-256), ...])  URL は上から順に試す
#   出典: OpenCV Zoo https://github.com/opencv/opencv_zoo （Apache License 2.0）
ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
HF = "https://huggingface.co/opencv/"
OPENCV_MODELS = [
    ("person_detection_mediapipe_2023mar.onnx", [
        (HF + "person_detection_mediapipe/resolve/main/person_detection_mediapipe_2023mar.onnx",
         "47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f"),
        (ZOO + "8e2b658d1007864a24cc2cf0e5b4288c74b32584/models/person_detection_mediapipe/"
         "person_detection_mediapipe_2023mar.onnx",
         "47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f")]),
    ("pose_estimation_mediapipe_2023mar.onnx", [
        (HF + "pose_estimation_mediapipe/resolve/main/pose_estimation_mediapipe_2023mar.onnx",
         "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f"),
        (ZOO + "1f19f821d68288feff2ef5c53993b33da74b1509/models/pose_estimation_mediapipe/"
         "pose_estimation_mediapipe_2023mar.onnx",
         "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f")]),
]

# 出典: MediaPipe https://ai.google.dev/edge/mediapipe/solutions/guide
#   latest は更新されることがあるので、SHA-256 が合わなければ固定版（/1/）を使う
MP = "https://storage.googleapis.com/mediapipe-models/"
MEDIAPIPE_MODELS = [
    ("face_landmarker.task", [
        (MP + "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
         "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
        (MP + "face_landmarker/face_landmarker/float16/1/face_landmarker.task",
         "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff")]),
    ("pose_landmarker_full.task", [
        (MP + "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
         "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad"),
        (MP + "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
         "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1")]),
    ("hand_landmarker.task", [
        (MP + "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
         "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"),
        (MP + "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
         "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1")]),
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ssl_contexts():
    """通常の証明書 → pip に同梱の証明書 の順に試す
    （macOS の python.org 版 Python では証明書が未設定のことがあるため）"""
    yield ssl.create_default_context()
    try:
        from pip._vendor import certifi
        yield ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass


def download(url, dest):
    last_error = None
    for ctx in ssl_contexts():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cv-lecture-samples"})
            with urllib.request.urlopen(req, context=ctx, timeout=60) as r, open(dest, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done, shown = 0, -1
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    pct = done * 100 // total if total else 0
                    if total and pct // 5 != shown:      # 5% ごとに表示
                        shown = pct // 5
                        print("\r    %5.1f / %5.1f MB (%3d%%)" % (done / 1e6, total / 1e6, pct),
                              end="", flush=True)
            print()
            return
        except ssl.SSLError as e:
            last_error = e          # 次の証明書で再試行
        except Exception as e:
            raise e
    raise last_error


def fetch(name, sources):
    """1 つのモデルをダウンロードする。成功したら True"""
    path = os.path.join(MODEL_DIR, name)
    known = {sha for _, sha in sources}
    if os.path.exists(path) and sha256_of(path) in known:
        print("[OK] %s（ダウンロード済み）" % name)
        return True
    print("ダウンロード中: %s" % name)
    tmp = path + ".part"
    for url, sha in sources:
        try:
            download(url, tmp)
            if sha256_of(tmp) != sha:
                raise ValueError("ファイルの内容が想定と違います（SHA-256 不一致）")
            os.replace(tmp, path)
            print("[OK] %s" % name)
            return True
        except Exception as e:
            print("    失敗: %s (%s)" % (url.split("/")[2], e))
            if os.path.exists(tmp):
                os.remove(tmp)
    return False


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    models = list(OPENCV_MODELS)
    if importlib.util.find_spec("mediapipe") is not None:
        models += MEDIAPIPE_MODELS
    else:
        print("（mediapipe がインストールされていないので、mediapipe_demo.py 用のモデルは省略します）")
    failed = [name for name, sources in models if not fetch(name, sources)]
    if failed:
        print()
        print("ダウンロードできなかったモデルがあります: " + ", ".join(failed))
        print("ネットワークを確認して、もう一度 python download_models.py を実行してください。")
        print("（ほかの機能はこのままでも使えます）")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
