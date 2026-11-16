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

キー操作（画像ウィンドウを選択した状態で）:
    t   : カメラ画像からテンプレートを選び直す
    ESC / q : 終了
"""

import argparse
import sys

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
    parser.add_argument("--scale", type=float, default=0.5,
                        help="処理のためにカメラ画像を縮小する倍率 (default: 0.5)")
    # 互換のため、オプションなしの数字もカメラ番号として受け付ける（python flann.py 1）
    parser.add_argument("camera_pos", nargs="?", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cam = args.camera_pos if args.camera_pos is not None else args.camera

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

    print("t: テンプレートを選び直す   ESC/q: 終了")

    while True:
        ret, img2 = cap.read()
        if not ret:
            print("image not captured. stop.")
            break
        img2 = cv2.resize(img2, None, fx=args.scale, fy=args.scale)
        kp2, des2 = detector.detectAndCompute(img2, None)

        good = []
        matchesMask = None
        if des1 is not None and des2 is not None and len(kp2) >= 2:
            # 各テンプレート特徴点について、カメラ画像中の近い候補を2つ探す
            matches = matcher.knnMatch(des1, des2, k=2)
            good = ratio_test(matches, args.ratio)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # RANSAC 等で射影変換行列を推定（再投影誤差 5 画素以内をインライアとする）
            M, mask = cv2.findHomography(src_pts, dst_pts, method, 5.0)
            if M is not None:
                matchesMask = mask.ravel().tolist()
                if sum(matchesMask) >= 4:
                    # テンプレートの四隅を射影変換してカメラ画像上に描く
                    h, w = img1.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    img2 = cv2.polylines(img2, [np.int32(dst)], True, (255, 255, 0), 3, cv2.LINE_AA)

        draw_params = dict(matchColor=(0, 255, 0),     # 対応（インライア）を緑で表示
                           singlePointColor=None,
                           matchesMask=matchesMask,     # インライアのみ描画
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        n_in = sum(matchesMask) if matchesMask else 0
        cv2.putText(img3, "%s/%s  matches: %d  inliers: %d"
                    % (args.feature, args.method, len(good), n_in), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        img3 = cv2.resize(img3, None, fx=1.0 / args.scale, fy=1.0 / args.scale)
        cv2.imshow("detect", img3)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('t'):
            cv2.destroyWindow("detect")
            img1 = select_template_from_camera(cap, args.scale)
            kp1, des1 = prepare(img1)

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
