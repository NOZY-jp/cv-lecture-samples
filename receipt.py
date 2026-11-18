# coding: utf-8
"""
実行記録（提出用レシート）

課題で「自分の環境で実際に実行したこと」を確認するための仕組みです。

    python capture.py --task 課題コード            # 学籍番号を聞かれます
    python capture.py --task 課題コード --id 学籍番号

で起動すると、画面の左下にあなた用の課題（使う処理など）が表示されます。
課題をこなしてから q で終了すると（または R キーで）、receipts/ フォルダに
記録ファイルが保存され、確認コードが表示されます。記録ファイルを提出してください。

記録に含まれるもの（画像は一切含みません）:
  ・学籍番号、課題コード、日時
  ・この環境のインストール ID（初回の実行時に乱数で作られ、このフォルダと
    ユーザーの設定フォルダの 2 か所に保存されます）
  ・PC の識別情報をハッシュにしたもの（元の情報には戻せません）
  ・OS・Python・OpenCV の版、カメラの解像度、FPS
  ・使った処理とフレーム数、処理中の数値の要約（入力画像の明るさ、二値化のしきい値など）

注意: 同じ環境（フォルダのコピーや同じ PC）で複数の人の記録を作ると、提出物から分かります。
"""

import base64
import datetime
import hashlib
import hmac
import json
import os
import platform
import sys
import time
import uuid
import zlib

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RECEIPT_DIR = os.path.join(HERE, "receipts")
LOCAL_ID_FILE = os.path.join(HERE, ".install_id")
FORMAT_VERSION = 1

# 記録の改ざんや写し間違いを見つけるための鍵（秘密ではありません）
_CHECK_KEY = b"cv-lecture-samples/receipt/v1"
# PC の識別情報をハッシュにするときに混ぜる値
_MACHINE_SALT = b"cv-lecture-samples/machine/v1"

# 課題で使うよう指定する処理の候補（追加のダウンロードが不要なもの）
TASK_MODE_CANDIDATES = ["h", "S", "V", "x", "y", "s", "a", ":", "f", "l", ";", "d", "e", "E",
                        "F", "Y"]
TASK_MIN_FRAMES = 60        # 指定した処理を使うフレーム数
TASK_HOLD_FRAMES = 30       # 指定したしきい値で二値化を続けるフレーム数
TASK_TOLERANCE = 1          # しきい値の許容差（矢印キーは 2 ずつ動くので ±1 を認める）


def held_frames(manual, target, tolerance=TASK_TOLERANCE):
    """手動しきい値ごとのフレーム数 {"137": 40, ...} から、target±tolerance で続けたフレーム数"""
    return sum(n for t, n in manual.items() if abs(int(t) - target) <= tolerance)


# ----------------------------------------------------------------------
# インストール ID
# ----------------------------------------------------------------------
def _user_id_file():
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "cv-lecture-samples", "install_id")


def _read_id(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("id"), str) and len(data["id"]) == 32:
            return data
    except (OSError, ValueError):
        pass
    return None


def _write_id(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return True
    except OSError:
        return False


def ensure_install_id():
    """インストール ID を用意する（なければ作る）。
    このフォルダとユーザーの設定フォルダの 2 か所に保存し、片方が消えたらもう片方から戻す。
    2 か所の ID が食い違う（例: 他人のフォルダをコピーした）ときは、両方をそのまま記録する。"""
    user_file = _user_id_file()
    local, user = _read_id(LOCAL_ID_FILE), _read_id(user_file)
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    if local is None and user is None:
        data = {"id": uuid.uuid4().hex, "created": now}
        _write_id(LOCAL_ID_FILE, data)
        _write_id(user_file, data)
        local = user = data
        event = "created"
    elif local is None:
        _write_id(LOCAL_ID_FILE, user)
        local = user
        event = "restored_local"
    elif user is None:
        _write_id(user_file, local)
        user = local
        event = "restored_user"
    else:
        event = "ok" if local["id"] == user["id"] else "mismatch"
    return {"local_id": local["id"], "local_created": local.get("created"),
            "user_id": user["id"], "user_created": user.get("created"), "event": event}


def machine_hash():
    """PC の識別情報（ネットワーク機器の固有番号・ホスト名・ユーザー名）のハッシュ。元には戻せない"""
    node = uuid.getnode()
    reliable = not (node >> 40) & 1          # 固有番号が取れず乱数になったときは 1 が立つ
    try:
        user = os.getlogin()
    except OSError:
        user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    raw = ("%012x|%s|%s" % (node, platform.node(), user)).encode("utf-8")
    return hashlib.sha256(_MACHINE_SALT + raw).hexdigest()[:16], bool(reliable)


# ----------------------------------------------------------------------
# 課題の割り当て（課題コードと学籍番号から決まる。教員側でも同じ計算で確認する）
# ----------------------------------------------------------------------
def normalize(text):
    return "".join(str(text).split()).upper()


def assign_tasks(task_code, student_id):
    h = hashlib.sha256(("%s|%s" % (normalize(task_code), normalize(student_id))).encode("utf-8")).digest()
    threshold = 70 + int.from_bytes(h[:2], "big") % 121              # 70〜190
    order = sorted(TASK_MODE_CANDIDATES, key=lambda k: hashlib.sha256(h + k.encode()).digest())
    return {"threshold": threshold, "modes": order[:3],
            "min_frames": TASK_MIN_FRAMES, "hold_frames": TASK_HOLD_FRAMES}


# ----------------------------------------------------------------------
# 記録
# ----------------------------------------------------------------------
class Recorder:
    def __init__(self, task_code, student_id, source, frame_size, key_of_proc):
        self.task_code = normalize(task_code)
        self.student_id = normalize(student_id)
        self.tasks = assign_tasks(self.task_code, self.student_id)
        self.source = source                    # "camera" または "file"
        self.frame_size = frame_size            # (幅, 高さ)
        self.key_of_proc = key_of_proc          # 処理名 → キー
        self.install = ensure_install_id()
        self.machine, self.machine_reliable = machine_hash()
        self.started = time.time()
        self.prev_t = None
        self.frames = 0
        self.modes = {}
        self.saved_path = None

    def _mode(self, proc):
        if proc not in self.modes:
            self.modes[proc] = {"frames": 0, "seconds": 0.0,
                                "bright_sum": 0.0, "bright_sq": 0.0, "bright_min": 255.0, "bright_max": 0.0}
        return self.modes[proc]

    def record(self, proc, img, st):
        """1 フレームごとに呼ぶ。img は処理前の入力画像"""
        t = time.time()
        dt = 0.0 if self.prev_t is None else min(t - self.prev_t, 1.0)
        self.prev_t = t
        self.frames += 1
        m = self._mode(proc)
        m["frames"] += 1
        m["seconds"] += dt
        b = float(cv2.cvtColor(cv2.resize(img, (64, 48)), cv2.COLOR_BGR2GRAY).mean())
        m["bright_sum"] += b
        m["bright_sq"] += b * b
        m["bright_min"] = min(m["bright_min"], b)
        m["bright_max"] = max(m["bright_max"], b)
        if proc == "binary":
            otsu = int(getattr(st, "bin_last_otsu", -1))
            m.setdefault("otsu", {"min": 255, "max": 0, "sum": 0})
            m["otsu"]["min"] = min(m["otsu"]["min"], otsu)
            m["otsu"]["max"] = max(m["otsu"]["max"], otsu)
            m["otsu"]["sum"] += otsu
            manual = getattr(st, "bin_manual", None)
            if manual is not None:
                held = m.setdefault("manual", {})
                held[str(manual)] = held.get(str(manual), 0) + 1

    # --- 課題の達成状況 ---
    def progress(self, japanese=True):
        """[(説明, 達成したか), ...]"""
        tk = self.tasks
        held = held_frames(self.modes.get("binary", {}).get("manual", {}), tk["threshold"])
        n, need = min(held, tk["hold_frames"]), tk["hold_frames"]
        text = ("b: しきい値を手動で %d（±%d）にする（%d/%d）" if japanese
                else "b: set threshold to %d (+-%d) manually (%d/%d)") % (
            tk["threshold"], TASK_TOLERANCE, n, need)
        items = [(text, held >= need)]
        for key in tk["modes"]:
            proc = next((p for p, k in self.key_of_proc.items() if k == key), key)
            count = self.modes.get(proc, {}).get("frames", 0)
            need = tk["min_frames"]
            text = ("%s: %s を使う（%d/%d）" if japanese else "%s: use %s (%d/%d)") % (
                key, proc, min(count, need), need)
            items.append((text, count >= need))
        return items

    # --- 保存 ---
    def payload(self):
        modes = {}
        for proc, m in self.modes.items():
            n = max(m["frames"], 1)
            mean = m["bright_sum"] / n
            entry = {"frames": m["frames"], "seconds": round(m["seconds"], 2),
                     "bright_mean": round(mean, 2),
                     "bright_std": round(max(m["bright_sq"] / n - mean * mean, 0.0) ** 0.5, 2),
                     "bright_min": round(m["bright_min"], 2), "bright_max": round(m["bright_max"], 2)}
            if "otsu" in m:
                entry["otsu"] = {"min": m["otsu"]["min"], "max": m["otsu"]["max"],
                                 "mean": round(m["otsu"]["sum"] / n, 2)}
            if "manual" in m:
                entry["manual"] = m["manual"]
            modes[proc] = entry
        elapsed = time.time() - self.started
        return {
            "version": FORMAT_VERSION,
            "student_id": self.student_id,
            "task_code": self.task_code,
            "tasks": self.tasks,
            "progress": [[text, done] for text, done in self.progress()],
            "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "install": self.install,
            "machine": {"hash": self.machine, "reliable": self.machine_reliable},
            "env": {"os": platform.platform(), "python": platform.python_version(),
                    "opencv": cv2.__version__, "numpy": np.__version__},
            "session": {"source": self.source, "frame_size": list(self.frame_size),
                        "frames": self.frames, "seconds": round(elapsed, 2),
                        "fps": round(self.frames / max(elapsed, 1e-6), 2)},
            "modes": modes,
        }

    def save(self):
        """記録ファイルを保存し、(ファイルのパス, 確認コード) を返す"""
        data = self.payload()
        body = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        sig = hmac.new(_CHECK_KEY, body.encode("utf-8"), hashlib.sha256).hexdigest()
        blob = base64.b64encode(zlib.compress(json.dumps({"data": data, "sig": sig},
                                                        ensure_ascii=False).encode("utf-8"), 9)).decode()
        code = "%s-%s-%s" % (sig[0:4].upper(), sig[4:8].upper(), sig[8:12].upper())
        os.makedirs(RECEIPT_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(RECEIPT_DIR, "receipt_%s_%s_%s.txt" % (self.student_id, self.task_code, stamp))
        lines = ["# 実行記録（このファイルを提出してください）",
                 "# 学籍番号: %s   課題コード: %s   作成: %s" % (self.student_id, self.task_code, data["created"]),
                 "# 確認コード: %s" % code,
                 "# 課題の達成状況:"]
        lines += ["#   [%s] %s" % ("済" if done else "  ", text) for text, done in self.progress()]
        lines += ["# 以下は記録の本体です（書き換えると無効になります）",
                  "-----BEGIN RECEIPT-----"]
        lines += [blob[i:i + 76] for i in range(0, len(blob), 76)]
        lines += ["-----END RECEIPT-----", ""]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self.saved_path = path
        return path, code


def decode_receipt(text):
    """記録ファイルの本文から (データ, 署名が正しいか) を返す（教員用の確認スクリプトから使う）"""
    inside = text.split("-----BEGIN RECEIPT-----", 1)[1].split("-----END RECEIPT-----", 1)[0]
    obj = json.loads(zlib.decompress(base64.b64decode("".join(inside.split()))).decode("utf-8"))
    data, sig = obj["data"], obj["sig"]
    body = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    ok = hmac.compare_digest(sig, hmac.new(_CHECK_KEY, body.encode("utf-8"), hashlib.sha256).hexdigest())
    return data, ok, sig


def ask_student_id():
    while True:
        sid = normalize(input("学籍番号を入力してください: "))
        if sid and all(c.isalnum() or c in "-_" for c in sid):
            return sid
        print("英数字で入力してください。")
