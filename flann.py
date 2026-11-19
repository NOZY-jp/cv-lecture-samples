# coding: utf-8
"""
特徴点マッチング＋RANSAC による平面物体の検出

テンプレート画像（本の表紙・ポスターなどの平面物体）とカメラ画像の間で
局所特徴（SIFT など）を抽出し、FLANN による近傍探索で対応付けたあと、
RANSAC で射影変換行列（ホモグラフィ）を推定して、カメラ画像中の
テンプレートの位置を四角形で表示します。

使い方:
    python flann.py                          # カメラ画像からテンプレートを範囲選択
    python flann.py --template book.jpg      # 画像ファイルをテンプレートにする
    python flann.py --camera 1               # カメラ番号を指定
    python flann.py --feature akaze          # 特徴量を変更 (sift/surf/kaze/akaze/brisk/orb)
    python flann.py --method magsac          # ロバスト推定法を変更 (ransac/magsac)
    python flann.py --reproj 3               # インライアとみなす再投影誤差の閾値 [画素]
    python flann.py --task 課題コード         # 課題の実行記録をとる（receipt.py を参照）

キー操作（画像ウィンドウを選択した状態で）:
    o   : アウトライア（RANSAC で外れ値とされた対応）の表示 ON/OFF
    t   : カメラ画像からテンプレートを選び直す
    R   : 実行記録を保存（--task で起動したとき）
    ESC / q : 終了

対応点の線の色:
    緑〜オレンジ: インライア。再投影誤差（テンプレート側の点を推定した射影変換で写した位置と、
                  実際に対応したカメラ画像側の点との距離）が小さいほど緑、閾値に近いほどオレンジ
    赤: アウトライア（ratio test は通ったが、誤差が閾値を超え外れ値とされた対応）
    黄: 対応が少なく射影変換を推定しなかったときの、ratio test を通った対応
"""

import argparse
import sys
import types

import cv2
import numpy as np

MIN_MATCH_COUNT = 10        # ホモグラフィを推定するのに必要な対応点の最小数
TEMPLATE_MAX_SIZE = 640     # テンプレート画像が大きすぎるときはこのサイズまで縮小


# ----------------------------------------------------------------------
# 特徴量抽出器とマッチャーの作成
# ----------------------------------------------------------------------
# 特徴量名: (生成関数名, 引数, 記述子の種類)
#   OpenCV 5 では SIFT / ORB は本体、KAZE / AKAZE / BRISK / SURF は contrib 版の
#   cv2.xfeatures2d に移動した（OpenCV 4 では SURF 以外は cv2 直下）
FEATURES = {
    "sift":  ("SIFT_create",  {},                  "float"),
    "surf":  ("SURF_create",  {"hessianThreshold": 400}, "float"),
    "kaze":  ("KAZE_create",  {},                  "float"),
    "akaze": ("AKAZE_create", {},                  "binary"),
    "brisk": ("BRISK_create", {},                  "binary"),
    "orb":   ("ORB_create",   {"nfeatures": 2000}, "binary"),
}

# 射影変換のロバスト推定法
METHODS = {
    "ransac": "RANSAC",         # 古典的な RANSAC
    "magsac": "USAC_MAGSAC",    # MAGSAC++（OpenCV 4.5 以降の USAC フレームワーク）
}


def find_cv_function(name):
    """cv2 直下 → cv2.xfeatures2d の順に関数を探す"""
    for mod in (cv2, getattr(cv2, "xfeatures2d", None)):
        if mod is not None and hasattr(mod, name):
            return getattr(mod, name)
    return None


def create_detector_and_matcher(name):
    func_name, kwargs, desc_type = FEATURES[name]
    factory = find_cv_function(func_name)
    if factory is None:
        print("この OpenCV (%s) では %s が使えません。" % (cv2.__version__, name.upper()))
        print("opencv-contrib-python を入れてください（README.md 参照）。")
        sys.exit(1)
    try:
        detector = factory(**kwargs)
    except cv2.error:
        # SURF は特許の関係で、pip 版では無効化されている（自前ビルドが必要）
        print("%s はこの OpenCV ではビルド時に無効化されています（特許アルゴリズム）。" % name.upper())
        sys.exit(1)

    if desc_type == "float":
        # 実数ベクトルの特徴量 → kd-tree による近傍探索
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    else:
        # バイナリ特徴量 → LSH (Locality Sensitive Hashing) による近傍探索
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    return detector, matcher


# ----------------------------------------------------------------------
# テンプレートの準備
# ----------------------------------------------------------------------
def shrink(img, max_size):
    h, w = img.shape[:2]
    s = max_size / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def load_template(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("テンプレート画像を読み込めませんでした: %s" % path)
        sys.exit(1)
    return shrink(img, TEMPLATE_MAX_SIZE)


def select_template_from_camera(cap, scale):
    """カメラ映像を表示し、スペースキーで静止 → マウスで範囲を選択"""
    win = "select template"
    print("-" * 60)
    print("検出したい平面物体（本の表紙など模様の多いもの）をカメラに映し、")
    print("スペースキーを押して静止させてください。（ESC/q で終了）")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("カメラから画像を取得できませんでした。")
            sys.exit(1)
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        view = frame.copy()
        cv2.putText(view, "SPACE: freeze & select", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            sys.exit(0)
        if key == ord(' '):
            print("マウスでドラッグして範囲を選び、Enter（またはスペース）で決定、")
            print("c でやり直します。")
            x, y, w, h = cv2.selectROI(win, frame, showCrosshair=False)
            if w > 0 and h > 0:
                cv2.destroyWindow(win)
                cv2.waitKey(1)
                roi = frame[y:y + h, x:x + w]
                return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            print("範囲が選択されませんでした。もう一度スペースキーを押してください。")


# ----------------------------------------------------------------------
# マッチングと射影変換の推定
# ----------------------------------------------------------------------
def ratio_test(knn_matches, ratio):
    """Lowe の ratio test: 1番目と2番目に近い候補の距離比で良い対応だけ残す"""
    good = []
    for pair in knn_matches:
        if len(pair) < 2:       # 候補が2つ見つからない場合がある（LSH など）
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


COLOR_OUTLIER = (0, 0, 255)      # 赤
COLOR_CANDIDATE = (0, 255, 255)  # 黄


def reprojection_errors(H, src_pts, dst_pts):
    """各対応の再投影誤差 [画素]:
    テンプレート側の点を射影変換 H で写した位置と、カメラ画像側の対応点との距離"""
    projected = cv2.perspectiveTransform(src_pts, H)
    return np.linalg.norm(projected - dst_pts, axis=2).ravel()


def error_color(err, threshold):
    """再投影誤差を色にする: 0 = 緑 → 閾値の半分 = 黄 → 閾値 = オレンジ"""
    t = min(max(err / threshold, 0.0), 1.0)
    hue = int(round(60 - 48 * t))           # OpenCV の色相（0〜179）: 60=緑, 30=黄, 12=オレンジ
    bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def side_by_side(img1, img2):
    """2 枚の画像を左右に並べる。(並べた画像, 右の画像の左端の x 座標) を返す"""
    a = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1
    b = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if img2.ndim == 2 else img2
    canvas = np.zeros((max(a.shape[0], b.shape[0]), a.shape[1] + b.shape[1], 3), np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1]:] = b
    return canvas, a.shape[1]


def draw_correspondences(img1, kp1, img2, kp2, good, inlier_mask, errors, threshold,
                         show_outliers=True, zoom=1.0):
    """対応点を線で結んだ画像を作る（zoom 倍に拡大してから描くので線がぼやけない）。
    inlier_mask, errors: good と同じ長さ（射影変換を推定しなかったときは None）"""
    canvas, offset = side_by_side(img1, img2)
    canvas = cv2.resize(canvas, None, fx=zoom, fy=zoom)

    def draw(m, color):
        p = kp1[m.queryIdx].pt
        q = kp2[m.trainIdx].pt
        p = (int(round(p[0] * zoom)), int(round(p[1] * zoom)))
        q = (int(round((q[0] + offset) * zoom)), int(round(q[1] * zoom)))
        cv2.line(canvas, p, q, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p, 3, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, q, 3, color, 1, cv2.LINE_AA)

    if inlier_mask is None:
        # 射影変換を推定していない → ratio test を通った対応をすべて黄色で
        for m in good:
            draw(m, COLOR_CANDIDATE)
        return canvas
    if show_outliers:                                   # アウトライアを先に赤で描く
        for m, inl in zip(good, inlier_mask):
            if not inl:
                draw(m, COLOR_OUTLIER)
    # インライアは誤差の小さい順に描く（誤差の大きい線が上に来て目立つ）
    inliers = [i for i, inl in enumerate(inlier_mask) if inl]
    for i in sorted(inliers, key=lambda i: errors[i]):
        draw(good[i], error_color(errors[i], threshold))
    return canvas


def draw_error_legend(img, threshold, x, y, width=200):
    """再投影誤差の色の凡例（カラーバー）と、アウトライアの色を描く"""
    for i in range(width):
        cv2.line(img, (x + i, y), (x + i, y + 12), error_color(threshold * i / (width - 1), threshold), 1)
    cv2.rectangle(img, (x, y), (x + width, y + 12), (0, 0, 0), 1)
    put_label(img, "0", (x - 4, y + 32), (255, 255, 255), 0.5)
    put_label(img, "%g px" % threshold, (x + width - 30, y + 32), (255, 255, 255), 0.5)
    put_label(img, "inlier error", (x + width // 2 - 45, y - 8), (255, 255, 255), 0.5)
    cv2.rectangle(img, (x + width + 20, y), (x + width + 40, y + 12), COLOR_OUTLIER, -1)
    put_label(img, "outlier", (x + width + 46, y + 11), COLOR_OUTLIER, 0.5)
    cv2.rectangle(img, (x + width + 130, y), (x + width + 150, y + 12), COLOR_CANDIDATE, -1)
    put_label(img, "candidate (no homography)", (x + width + 156, y + 11), COLOR_CANDIDATE, 0.5)


def put_label(img, text, org, color, scale=0.7):
    """黒いふち付きの文字（どんな背景でも読めるように）"""
    # ふちは同じ太さの黒い文字を少しずらして重ねて描く（太い黒い文字を下に敷く方法は、
    # OpenCV 5 では太さで文字の幅が変わるため、文字がずれてしまう）
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(img, text, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2,
                    cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="特徴点マッチング＋RANSACによる平面物体検出")
    parser.add_argument("--camera", "-c", type=int, default=0, help="カメラ番号 (default: 0)")
    parser.add_argument("--template", "-t", default=None,
                        help="テンプレート画像ファイル（省略するとカメラ画像から選択）")
    parser.add_argument("--feature", "-f", default="sift", choices=list(FEATURES),
                        help="局所特徴量の種類 (default: sift)")
    parser.add_argument("--method", "-m", default="ransac", choices=list(METHODS),
                        help="射影変換のロバスト推定法 (default: ransac)")
    parser.add_argument("--ratio", type=float, default=0.7, help="ratio test の閾値 (default: 0.7)")
    parser.add_argument("--reproj", type=float, default=5.0,
                        help="インライアとみなす再投影誤差の閾値 [縮小後の画像の画素] (default: 5)")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="処理のためにカメラ画像を縮小する倍率 (default: 0.5)")
    parser.add_argument("--task", default=None, help="課題コード（指定すると実行記録をとる）")
    parser.add_argument("--id", default=None, help="学籍番号（--task のとき。省略すると入力を求める）")
    # 互換のため、オプションなしの数字もカメラ番号として受け付ける（python flann.py 1）
    parser.add_argument("camera_pos", nargs="?", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cam = args.camera_pos if args.camera_pos is not None else args.camera

    # 実行記録用のインストール ID（初回の実行時に作られる）
    import receipt
    try:
        receipt.ensure_install_id()
    except Exception:
        pass
    recorder = None
    if args.task:
        student_id = receipt.normalize(args.id) if args.id else receipt.ask_student_id()

    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print("カメラ %d を開けませんでした。--camera で番号を変えて試してください。" % cam)
        sys.exit(1)

    detector, matcher = create_detector_and_matcher(args.feature)
    if not hasattr(cv2, METHODS[args.method]):
        print("この OpenCV (%s) では %s が使えません。" % (cv2.__version__, args.method))
        sys.exit(1)
    method = getattr(cv2, METHODS[args.method])
    print("feature: %s   method: %s" % (args.feature, args.method))

    def prepare(img1):
        kp1, des1 = detector.detectAndCompute(img1, None)
        print("template size: %dx%d, keypoints: %d" % (img1.shape[1], img1.shape[0], len(kp1)))
        if des1 is None or len(kp1) < MIN_MATCH_COUNT:
            print("テンプレートから特徴点が十分に得られませんでした。模様の多い物体を選んでください。")
        return kp1, des1

    if args.template:
        img1 = load_template(args.template)
    else:
        img1 = select_template_from_camera(cap, args.scale)
    kp1, des1 = prepare(img1)

    mode = "flann:%s/%s" % (args.feature, args.method)          # 実行記録での処理名
    if args.task:
        w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        recorder = receipt.Recorder(args.task, student_id, "camera", (w0, h0), script="flann")
        print("課題コード %s / 学籍番号 %s で実行記録をとります（R: 保存）。" % (recorder.task_code, recorder.student_id))
    print("o: アウトライアの表示 ON/OFF   t: テンプレートを選び直す   ESC/q: 終了")
    print("線の色  緑〜オレンジ: インライア（再投影誤差が小さいほど緑）  赤: アウトライア  黄: RANSAC 前の候補")
    show_outliers = True

    while True:
        ret, img2 = cap.read()
        if not ret:
            print("image not captured. stop.")
            break
        img2 = cv2.resize(img2, None, fx=args.scale, fy=args.scale)
        raw = img2.copy() if recorder is not None else None
        kp2, des2 = detector.detectAndCompute(img2, None)

        good = []
        matchesMask, errors = None, None
        if des1 is not None and des2 is not None and len(kp2) >= 2:
            # 各テンプレート特徴点について、カメラ画像中の近い候補を2つ探す
            matches = matcher.knnMatch(des1, des2, k=2)
            good = ratio_test(matches, args.ratio)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # RANSAC 等で射影変換行列を推定（再投影誤差が閾値以内の対応をインライアとする）
            M, mask = cv2.findHomography(src_pts, dst_pts, method, args.reproj)
            if M is not None:
                matchesMask = mask.ravel().tolist()
                errors = reprojection_errors(M, src_pts, dst_pts)
                if sum(matchesMask) >= 4:
                    # テンプレートの四隅を射影変換してカメラ画像上に描く
                    h, w = img1.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    img2 = cv2.polylines(img2, [np.int32(dst)], True, (255, 255, 0), 3, cv2.LINE_AA)

        img3 = draw_correspondences(img1, kp1, img2, kp2, good, matchesMask, errors, args.reproj,
                                    show_outliers, zoom=1.0 / args.scale)
        put_label(img3, "%s/%s  matches: %d" % (args.feature, args.method, len(good)),
                  (10, 30), (255, 255, 255))
        # 表示が点滅しないよう、射影変換を推定できなかったフレームでも同じ行構成にする
        hidden = "" if show_outliers else " (hidden, o)"
        if matchesMask is None:
            put_label(img3, "inliers: -  (no homography: %d candidates)" % len(good), (10, 60),
                      COLOR_CANDIDATE)
            put_label(img3, "outliers: -" + hidden, (10, 90), COLOR_OUTLIER)
        else:
            inlier_err = errors[np.array(matchesMask, bool)]
            n_in = len(inlier_err)
            stats = ("  error mean %.2f / max %.2f px" % (inlier_err.mean(), inlier_err.max())
                     if n_in else "")
            put_label(img3, "inliers: %d%s" % (n_in, stats), (10, 60), error_color(0, 1))
            put_label(img3, "outliers: %d%s" % (len(good) - n_in, hidden), (10, 90), COLOR_OUTLIER)
        draw_error_legend(img3, args.reproj, 20, img3.shape[0] - 40)    # 凡例は常に表示
        if recorder is not None:
            detected = matchesMask is not None and sum(matchesMask) >= 4
            n_in = sum(matchesMask) if matchesMask else 0
            err = float(errors[np.array(matchesMask, bool)].mean()) if n_in else None
            recorder.record(mode, raw, types.SimpleNamespace(found=1 if detected else 0),
                            values={"matches": len(good), "inliers": n_in, "error_mean": err})
            rec = "REC %s / %s  (R: save)" % (recorder.task_code, recorder.student_id)
            (tw, _), _ = cv2.getTextSize(rec, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            put_label(img3, rec, (img3.shape[1] - tw - 12, 30), (80, 80, 255), 0.6)
        cv2.imshow("detect", img3)

        key = cv2.waitKey(1) & 0xFF
        if recorder is not None and key != 255:
            recorder.on_key(chr(key), mode)
        if key in (27, ord('q')):
            break
        elif key == ord('R') and recorder is not None:
            save_receipt(recorder)
        elif key == ord('o'):
            show_outliers = not show_outliers
        elif key == ord('t'):
            cv2.destroyWindow("detect")
            img1 = select_template_from_camera(cap, args.scale)
            kp1, des1 = prepare(img1)

    if recorder is not None:
        save_receipt(recorder)
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def save_receipt(recorder):
    path, code = recorder.save()
    print("-" * 60)
    print("実行記録を保存しました: %s" % path)
    print("確認コード: %s" % code)
    print("このファイルを提出してください。")
    print("-" * 60)


if __name__ == "__main__":
    main()
