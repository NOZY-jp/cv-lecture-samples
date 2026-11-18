# coding: utf-8
"""
実行記録（提出用）

課題で「自分の環境で実際に実行して、いろいろな機能を試したこと」を記録する仕組みです。

    python capture.py --task 課題コード            # 学籍番号を聞かれます
    python capture.py --task 課題コード --id 学籍番号

で起動すると、実行中の操作が記録されます（画面の左下に「記録中」と表示されます）。
q で終了すると（または途中で R キーを押すと）、receipts/ フォルダに記録ファイルが保存され、
確認コードが表示されます。記録ファイルを提出してください。

記録に含まれるもの（画像は一切含みません）:
  ・学籍番号、課題コード、日時
  ・この環境のインストール ID（初回の実行時に乱数で作られ、このフォルダと
    ユーザーの設定フォルダの 2 か所に保存されます）
  ・PC の識別情報をハッシュにしたもの（元の情報には戻せません）
  ・OS・Python・OpenCV の版、カメラの解像度、FPS
  ・処理ごとのフレーム数・時間・切り替えた回数、キー操作の回数、
    処理中の数値の要約（入力画像の明るさ、二値化のしきい値、顔や骨格が見つかったフレーム数など）

記録ファイルの本体は、記録（JSON）を圧縮して文字列にしたものに、改ざんを見つけるための
検査用の値を付けたものです。書き換えると無効になります。

注意: 同じ環境（フォルダのコピーや同じ PC）で複数の人の記録を作ると、提出物から分かります。
"""

import base64
import datetime
import hashlib
import hmac
import json
import os
import platform
import time
import uuid
import zlib

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RECEIPT_DIR = os.path.join(HERE, "receipts")
LOCAL_ID_FILE = os.path.join(HERE, ".install_id")
FORMAT_VERSION = 3

# 記録の改ざんや写し間違いを見つけるための鍵（秘密ではありません）
_CHECK_KEY = b"cv-lecture-samples/receipt/v1"
# PC の識別情報をハッシュにするときに混ぜる値
_MACHINE_SALT = b"cv-lecture-samples/machine/v1"

USED_MIN_FRAMES = 30        # 画面の「使った機能」に数えるフレーム数


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
# 記録
# ----------------------------------------------------------------------
def normalize(text):
    return "".join(str(text).split()).upper()


class Recorder:
    """処理ごとの利用状況と、キー操作の回数を記録する"""

    def __init__(self, task_code, student_id, source, frame_size):
        self.task_code = normalize(task_code)
        self.student_id = normalize(student_id)
        self.source = source                    # "camera" または "file"
        self.frame_size = frame_size            # (幅, 高さ)
        self.install = ensure_install_id()
        self.machine, self.machine_reliable = machine_hash()
        self.started = time.time()
        self.started_iso = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        self.prev_t = None
        self.prev_proc = None
        self.frames = 0
        self.switches = 0                       # 処理を切り替えた回数（合計）
        self.modes = {}                         # 処理名 → 利用状況
        self.keys = {}                          # "キー@処理名" → 押した回数
        self.saves = 0

    def _mode(self, proc):
        if proc not in self.modes:
            self.modes[proc] = {"frames": 0, "seconds": 0.0, "entries": 0,
                                "bright_sum": 0.0, "bright_sq": 0.0, "bright_min": 255.0, "bright_max": 0.0}
        return self.modes[proc]

    def on_key(self, key, proc):
        """キーが押されたときに呼ぶ。proc は押した時点で実行中の処理"""
        if key is None:
            return
        name = {chr(27): "ESC", " ": "SPACE"}.get(key, key)
        k = "%s@%s" % (name, proc)
        self.keys[k] = self.keys.get(k, 0) + 1

    def record(self, proc, img, st):
        """1 フレームごとに呼ぶ。img は処理前の入力画像"""
        t = time.time()
        dt = 0.0 if self.prev_t is None else min(t - self.prev_t, 1.0)
        self.prev_t = t
        self.frames += 1
        m = self._mode(proc)
        if proc != self.prev_proc:
            m["entries"] += 1
            if self.prev_proc is not None:
                self.switches += 1
            self.prev_proc = proc
        m["frames"] += 1
        m["seconds"] += dt
        b = float(cv2.cvtColor(cv2.resize(img, (64, 48)), cv2.COLOR_BGR2GRAY).mean())
        m["bright_sum"] += b
        m["bright_sq"] += b * b
        m["bright_min"] = min(m["bright_min"], b)
        m["bright_max"] = max(m["bright_max"], b)
        found = getattr(st, "found", None)          # 顔・上半身・骨格の検出で見つけた数
        if found is not None:
            m["detect_frames"] = m.get("detect_frames", 0) + 1
            m["found_frames"] = m.get("found_frames", 0) + (1 if found > 0 else 0)
            m["found_total"] = m.get("found_total", 0) + int(found)
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

    def summary(self):
        """(使った機能の数, 切り替えた回数, 記録したフレーム数)"""
        used = sum(1 for m in self.modes.values() if m["frames"] >= USED_MIN_FRAMES)
        return used, self.switches, self.frames

    # --- 保存 ---
    def payload(self):
        modes = {}
        for proc, m in self.modes.items():
            n = max(m["frames"], 1)
            mean = m["bright_sum"] / n
            entry = {"frames": m["frames"], "seconds": round(m["seconds"], 2), "entries": m["entries"],
                     "bright_mean": round(mean, 2),
                     "bright_std": round(max(m["bright_sq"] / n - mean * mean, 0.0) ** 0.5, 2),
                     "bright_min": round(m["bright_min"], 2), "bright_max": round(m["bright_max"], 2)}
            if "otsu" in m:
                entry["otsu"] = {"min": m["otsu"]["min"], "max": m["otsu"]["max"],
                                 "mean": round(m["otsu"]["sum"] / n, 2)}
            if "manual" in m:
                entry["manual"] = m["manual"]
            for k in ("detect_frames", "found_frames", "found_total"):
                if k in m:
                    entry[k] = m[k]
            modes[proc] = entry
        elapsed = time.time() - self.started
        return {
            "version": FORMAT_VERSION,
            "student_id": self.student_id,
            "task_code": self.task_code,
            "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "install": self.install,
            "machine": {"hash": self.machine, "reliable": self.machine_reliable},
            "env": {"os": platform.platform(), "python": platform.python_version(),
                    "opencv": cv2.__version__, "numpy": np.__version__},
            "session": {"started": self.started_iso, "source": self.source, "frame_size": list(self.frame_size),
                        "frames": self.frames, "seconds": round(elapsed, 2), "switches": self.switches,
                        "fps": round(self.frames / max(elapsed, 1e-6), 2), "saves": self.saves + 1},
            "modes": modes,
            "keys": self.keys,
        }

    def save(self):
        """記録ファイルを保存し、(ファイルのパス, 確認コード) を返す"""
        data = self.payload()
        self.saves += 1
        body = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        sig = hmac.new(_CHECK_KEY, body.encode("utf-8"), hashlib.sha256).hexdigest()
        blob = base64.b64encode(zlib.compress(json.dumps({"data": data, "sig": sig},
                                                        ensure_ascii=False).encode("utf-8"), 9)).decode()
        code = "%s-%s-%s" % (sig[0:4].upper(), sig[4:8].upper(), sig[8:12].upper())
        os.makedirs(RECEIPT_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(RECEIPT_DIR, "receipt_%s_%s_%s.txt" % (self.student_id, self.task_code, stamp))
        used = sorted((p for p, m in self.modes.items() if m["frames"] >= USED_MIN_FRAMES),
                      key=lambda p: -self.modes[p]["frames"])
        lines = ["# 実行記録（このファイルを提出してください）",
                 "# 学籍番号: %s   課題コード: %s   作成: %s" % (self.student_id, self.task_code, data["created"]),
                 "# 確認コード: %s" % code,
                 "# 記録したフレーム数: %d   処理を切り替えた回数: %d" % (self.frames, self.switches),
                 "# 使った機能（%d フレーム以上）: %s" % (USED_MIN_FRAMES, ", ".join(used) if used else "なし"),
                 "# 以下は記録の本体です（書き換えると無効になります）",
                 "-----BEGIN RECEIPT-----"]
        lines += [blob[i:i + 76] for i in range(0, len(blob), 76)]
        lines += ["-----END RECEIPT-----", ""]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path, code


def decode_receipt(text):
    """記録ファイルの本文から (データ, 検査用の値が正しいか, 検査用の値) を返す（教員用の確認スクリプトから使う）"""
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
