<img src="images/interaction_lab_logo.png" alt="Interaction Laboratory" width="150" align="right">

# 画像認識 講義用サンプルコード

**立命館大学 情報理工学部「実世界情報処理」教材**

講義で動かして見せた Python + OpenCV のサンプルプログラムです。
自分のパソコンで動かして、パラメータを変えたり処理を書き換えたりしてみてください。

| ファイル | 内容 |
| --- | --- |
| `capture.py` | Webカメラの画像を1フレームずつ取り込み、キー操作で選んだ画像処理（フィルタ・エッジ・二値化・ラベリング・オプティカルフロー・顔検出など）を行って表示します |
| `flann.py` | SIFT などの局所特徴の対応付けから RANSAC で射影変換行列を推定し、カメラ画像中の平面物体（本の表紙など）を検出します |
| `setup.sh` / `setup.bat` | セットアップ用（仮想環境の作成とモジュールのインストール） |
| `mediapipe_demo.py` | MediaPipe による Face Mesh・顔の向き・表情・骨格・手の推定（mediapipe が必要・任意） |
| `pose_estimation.py` | 人体の骨格推定の処理（`capture.py` の `K` キーから使われる。単体でも実行可） |
| `check_env.py` | 環境チェック用（モジュールが入っているか、各アルゴリズムやカメラが使えるか） |
| `download_models.py` | 骨格推定用の学習済みモデルのダウンロード（セットアップ時に自動実行） |
| `models/` | 学習済みモデル（顔検出の Haar-like カスケード・YuNet、骨格推定の MediaPipe Pose） |

## 必要なもの

- Python 3.8 以上（3.10〜3.13 推奨）
- Webカメラ（ノートPC内蔵カメラで OK）
- git（なくても ZIP でダウンロードできます）

Python が入っているかは、ターミナル（Windows は「コマンドプロンプト」または「PowerShell」）で次のように確認できます。

```
python --version
```

（macOS / Linux では `python3 --version` の場合もあります。以下 `python` を `python3` に読み替えてください。）

## 1. ダウンロード

```
git clone https://github.com/NobutakaShimada/cv-lecture-samples.git
cd cv-lecture-samples
```

git を使わない場合は、GitHub のページの緑色の **Code** ボタン → **Download ZIP** でダウンロードして展開し、ターミナルでそのフォルダに移動（`cd`）してください。

## 2. セットアップ（最初の1回だけ）

このフォルダの中に専用の**仮想環境**（`.venv` フォルダ）を作り、必要なモジュールをその中にインストールします。
パソコン全体の Python には何もインストールしないので、**すでに OpenCV などが入っていても衝突しません**。

**macOS / Linux**（ターミナルで、このフォルダに移動してから）

```
sh setup.sh
```

**Windows**

エクスプローラーで `setup.bat` を**ダブルクリック**するか、コマンドプロンプトでこのフォルダに移動してから次を実行します。

```
setup.bat
```

`numpy` と `opencv-contrib-python`、可能なら `mediapipe`（任意）がインストールされ、学習済みモデル（合計約 40MB）がダウンロードされたあと、最後に環境チェック（`check_env.py`）の結果が表示されます。
`mediapipe` は Intel 製 CPU の Mac には対応していないため、インストールできなくてもそのまま進みます（`mediapipe_demo.py` 以外は使えます）。
すべて `[OK]` になれば準備完了です。`[--]` は参考情報なので気にしなくて構いません。
`[NG]` があれば、表示される指示に従ってください。

うまくいかないときは、仮想環境を作り直してみてください（`sh setup.sh --clean` / `setup.bat --clean`）。

> **なぜ opencv-contrib-python なのか**
> OpenCV 5 では、Haar-like 特徴による顔検出（`CascadeClassifier`）や KAZE / AKAZE / BRISK などの
> 古典的なアルゴリズムが本体（`opencv-python`）から外れ、拡張版（`opencv-contrib-python`）に移りました。
> この講義では原理を学ぶために古典的な手法も使うので、拡張版を入れています。
> 拡張版は本体の機能もすべて含んでいます。OpenCV 4.8 以降であれば 4.x でも 5.x でも動きます。

## 3. 仮想環境を有効にする（ターミナルを開くたびに毎回）

プログラムを実行する前に、次のコマンドで仮想環境を有効にします。

| 環境 | コマンド |
| --- | --- |
| macOS / Linux | `source .venv/bin/activate` |
| Windows（コマンドプロンプト） | `.venv\Scripts\activate` |
| Windows（PowerShell） | `.venv\Scripts\Activate.ps1` |

プロンプトの先頭に `(.venv)` と表示されれば OK です。
これを忘れると、パソコン全体の Python（OpenCV が入っていない、または古い版）で実行されてしまいます。
`python check_env.py` を実行すると、仮想環境で動いているかどうかを確認できます。

> PowerShell で「スクリプトの実行が無効」と言われたら、
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` を一度実行してから再度試すか、コマンドプロンプトを使ってください。

> **Anaconda を使っている人へ**: プロンプトに `(base)` と `(.venv)` の両方が表示されていても問題ありません。
> `(.venv)` が表示されていることを確認してください。

<details>
<summary>セットアップスクリプトを使わずに手動で行う場合</summary>

macOS / Linux:

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_env.py
```

Windows:

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_env.py
```

</details>

## 4. 実行

### capture.py — カメラ画像のリアルタイム処理

```
python capture.py          # カメラ 0 を使う
python capture.py 1        # 外付けカメラなど、別のカメラを使う
python capture.py movie.mp4   # カメラの代わりに動画ファイルを使う
```

画像ウィンドウをクリックして選択した状態で、以下のキーを押すと処理が切り替わります。
**←→** の付いた処理は、`←` / `→`（または `,` / `.`）でパラメータを変えられます（`ESC` で元に戻ります）。
いまのパラメータの値は、画面上部の表示（例: `bits:3`）に出ます。

| 分類 | キー | 処理 |
| --- | --- | --- |
| 色・濃淡 | `c` | カラー（そのまま） |
| | `g` | グレースケール（輝度 Y）＋ヒストグラム |
| | `v` | 色空間のチャネル表示 **←→**（RGB → YIQ → HSV → L\*a\*b\*） |
| | `h` / `S` / `V` | 色相 / 彩度 / 明度（HSV） |
| | `Q` | 量子化 **←→**（1 チャネルあたりのビット数: 8 〜 1） |
| | `Z` | 標本化 **←→**（ブロックの大きさ: 1 〜 32 画素） |
| ヒストグラム・領域 | `b` | 二値化（大津の方法）＋ヒストグラム **←→**（しきい値を手動に） |
| | `e` / `E` | ヒストグラム平坦化 / 適応的ヒストグラム平坦化（CLAHE）＋ヒストグラム |
| | `l` | ラベリング |
| | `;` | ラベリング＋面積・重心・主軸 **←→**（表示する最小面積） |
| 特徴・フィルタ | `x` / `y` | Sobel フィルタ（x 方向 / y 方向） |
| | `s` / `a` | 勾配強度 / 勾配方向（色）と強度 |
| | `d` | DoG フィルタ |
| | `:` | SIFT 特徴点 |
| | `H` | Hough 変換: 直線と投票空間のヒートマップ **←→**（直線とみなす票数） |
| | `O` | Hough 変換: 円 **←→**（円の中心とみなす票数） |
| 検出 | `F` | 顔検出（Haar-like 特徴・Viola-Jones 法） |
| | `Y` | 顔検出（YuNet・CNN） |
| | `U` | 上半身検出（Haar-like 特徴） |
| | `P` | 人物検出（HOG 特徴 + SVM） **←→**（SVM の判定のしきい値） |
| | `K` | 人体の骨格推定（MediaPipe Pose） |
| 動き | `f` | 密なオプティカルフロー（Farneback） |
| | `t` | 特徴点追跡（Lucas-Kanade） |
| 自作 | `u` | **自作の処理**（後述） |

前処理（ぼかし）:

| キー | 処理 |
| --- | --- |
| `G` | ガウシアンぼかし |
| `m` | メディアンフィルタ |
| `B` | バイラテラルフィルタ |
| `↑` / `+` | ぼかしを強く（DoG の大きさも変わります） |
| `↓` / `-` | ぼかしを弱く |
| `ESC` | ぼかしをリセット（あわせて、いまの処理のパラメータも元に戻ります） |

`h`・`S`・`V` で HSV 色空間の色相・彩度・明度を、`g` でグレースケール（輝度 Y = 0.299R + 0.587G + 0.114B、YIQ の Y）を表示できます。
明度 V は R・G・B の最大値なので、例えば純粋な赤は V では白（255）、グレースケールでは暗い灰色（76）になります。

#### 色空間・量子化・標本化

- `v` では、左上に元の画像、残りの 3 つに色空間の各チャネルをグレースケールで表示します。`←` / `→` で RGB・YIQ・HSV・L\*a\*b\* を切り替えます。
  YIQ の I・Q や L\*a\*b\* の a\*・b\* のように負の値をとるチャネルは、灰色（128）が 0 です。
- `Q` では、1 チャネルあたりのビット数を減らして（2^ビット数 段階）、階調が粗くなる様子を見られます。
- `Z` では、ブロックごとに 1 つの値にまとめて、解像度（標本の数）を下げた様子を見られます。

#### ヒストグラムと二値化

`g`・`b`・`e`・`E` では、右下にグレースケールのヒストグラム（横軸は左が 0 = 黒、右が 255 = 白）が表示されます（`i` キーで表示の ON/OFF）。
`e`・`E` では、平坦化したあとの画像のヒストグラムが表示されるので、`g` と見比べてみてください。

`b` の二値化では、灰色の棒は黒に、白の棒は白になる画素です。
白飛び・黒つぶれで 255 や 0 に画素が集中すると、その棒はマゼンタで示され、全体に占める割合が表示されます（ほかの棒が見やすいよう、高さは両端を除いた最大に合わせています）。
ヒストグラムが櫛の歯のようにところどころ欠けるのは、多くの Web カメラが輝度を 16〜235 の約 220 段階で送ってきて、それを 0〜255 に引き伸ばしているためです。
緑の線が大津の方法で求めたしきい値、赤の線が実際に使っているしきい値です。ふだんは大津の値をそのまま使います（自動）。
`←` / `→`（または `,` / `.`）を押すと手動になり、しきい値をその値に固定して動かせます。手動の間は、画像が変わって大津の値が動いても、しきい値は変わりません。`ESC` で自動に戻ります。
水色の曲線は大津の方法が最大にしている**クラス間分散**で、その山の位置が大津のしきい値になっています。

#### 図形の特徴・Hough 変換・人物検出

- `;` では、二値化して白い領域ごとに、面積 S、重心（×）、**モーメント**から求めた主軸の向き（deg）と細長さ（e = 長軸 / 短軸）を表示します。
  白い楕円は、モーメントから求めた「同じ広がりをもつ楕円」です。画像の半分を超える領域は背景とみなして除きます。
- `H` では、エッジ（Canny）の各点が、通りうる直線 (ρ, θ) すべてに投票します。右下のヒートマップがその**投票空間**で、
  1 つのエッジ点の投票は 1 本の正弦波の曲線になります。曲線が多く重なった明るい点（緑の丸）が、検出した直線です。
- `O` では、エッジの勾配の向きを使って円の中心に投票し、票の多い位置を円の中心とします。
- `P` では、64×128 画素の窓を動かしながら、窓内の **HOG 特徴**（勾配の向きのヒストグラム）を学習済みの **SVM** で「人／人でない」に分類します。
  全身が映っている必要があります。`←` / `→` で SVM の判定のしきい値を変えると、誤検出と見落としの関係が分かります。

その他: `i` ヒストグラム表示の ON/OFF、`p` 顔検出（`F`）中に横顔検出の ON/OFF、`k` 骨格推定（`K`）中に平滑化の ON/OFF、`?` ヘルプ表示の ON/OFF、`q` 終了

起動するとキー操作の一覧がウィンドウに重ねて表示されます（背景は暗くなります）。`?` キーで消したり表示したりできます。選択中の処理は黄色で表示されます。
OpenCV 5 では日本語、OpenCV 4 では英語で表示されます（OpenCV 4 は日本語の文字を描けないため）。起動時に表示したくない場合は、`capture.py` 先頭の `SHOW_HELP_AT_START` を `False` にしてください。

#### 顔検出の古典的手法と CNN を比べてみる

- `F`: **Haar-like 特徴 + AdaBoost + カスケード**（Viola-Jones 法, 2001）。正面顔（青）、目（緑）、横顔（赤）を検出します。
- `Y`: **YuNet**（小さな CNN による顔検出器, OpenCV 4.8 以降で標準搭載）。顔の枠に加えて、目・鼻・口の両端の5点と信頼度を表示します。

顔を傾けたり、横を向いたり、手で一部を隠したりして、2つの手法の違いを確かめてみてください。

#### 人体の骨格推定

`K` キーで、カメラに映った人の骨格（関節 33 点）を推定して表示します。左半身はオレンジ、右半身は水色で描かれます。
OpenCV 5 には骨格推定の機能そのものは含まれていないため、[OpenCV Zoo](https://github.com/opencv/opencv_zoo) で公開されている
MediaPipe（BlazePose）のモデルを OpenCV の DNN モジュールで動かしています。
処理の中身は `pose_estimation.py` にまとめてあり、MediaPipe Pose（ライブラリ）が内部で行っている処理と同じ構成です。

1. **切り出し位置を決める**: 最初のフレームは**人物検出**で人を探し、「腰の中心」と「全身の大きさと向きを表す点」を求める。
   2フレーム目以降は、前のフレームの結果から決める（**追跡**）ので、人物検出を毎回行わずに済む
2. **骨格推定**: 人物を囲む正方形を、体がまっすぐ立つ向きに回転させて切り出し、関節の位置を推定する。
   モデルは関節の座標と、関節ごとの「ありそうな場所」を表す画像（**ヒートマップ**）の両方を出力するので、
   ヒートマップを使って座標を細かく補正する（特に顔の点の位置が正確になります）
3. **平滑化**: 推定結果は毎フレーム少しずつ揺れるので、**One Euro Filter** で時間方向になめらかにする。
   ゆっくり動くときは強くならして揺れを消し、速く動くときは弱くならして遅れを防ぐフィルタです
4. 平滑化した結果から、次のフレームの切り出し位置を決める（1 へ戻る）

実行中に `k` キーを押すと平滑化を OFF にできます。ON と OFF で骨格の揺れ方を比べてみてください。

カメラから少し離れて、**少なくとも肩から腰まで（できれば全身）が映るように**してください。顔のアップでは検出できません。
また、1人だけを対象にしています。

骨格推定だけを試したいときは、`python pose_estimation.py` で単体でも動かせます（`s` で平滑化の ON/OFF、`q` で終了）。
モデルはセットアップ時に自動でダウンロードされます。失敗した場合は `python download_models.py` を実行してください。

#### 自分で処理を書いてみる

`capture.py` の中の `proc_custom` 関数を書き換えて、実行中に `u` キーを押すと自分の処理が実行されます。
入力 `img` は BGR カラー画像（NumPy 配列）で、カラー画像またはグレースケール画像を返せば表示されます。

```python
def proc_custom(img, st):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges
```

新しいキーに割り当てたい場合は、`PROCS` という辞書に1行追加してください。

### flann.py — 特徴点マッチングと RANSAC による平面物体検出

```
python flann.py                        # カメラ画像からテンプレートを選ぶ
python flann.py --template book.jpg    # 画像ファイルをテンプレートにする
python flann.py --feature akaze        # 特徴量を変える
python flann.py --method magsac        # ロバスト推定法を変える
```

1. 起動するとカメラ映像が表示されます。本の表紙やお菓子の箱など、**模様の多い平らな物体**を映してスペースキーを押します。
2. 静止画上でマウスをドラッグして物体を囲み、Enter で決定します（`c` でやり直し）。
3. 物体を動かしたり傾けたりすると、検出された位置が水色の四角で表示されます。

対応点を結ぶ線の色の意味:

| 色 | 意味 |
| --- | --- |
| 緑 〜 黄 〜 オレンジ | **インライア**: RANSAC で推定した射影変換に合う対応。**再投影誤差**が小さいほど緑、閾値に近いほどオレンジ |
| 赤 | **アウトライア**: ratio test は通ったが、再投影誤差が閾値を超え、外れ値として除外された対応 |
| 黄（すべての線） | 対応点が少なく射影変換を推定しなかったときの、ratio test を通った対応 |

**再投影誤差**とは、テンプレート側の特徴点を推定した射影変換で写した位置と、実際に対応したカメラ画像側の特徴点との距離です。
左上にはインライアの再投影誤差の平均と最大が表示されます。インライアの数が多く、誤差が小さいほど、射影変換がよく当てはまっています。
（誤差は処理のために縮小した画像上の画素数です）

`o` キーでアウトライア（赤）の表示を消したり戻したりできます。`--method magsac`・`--ratio`・`--reproj` を変えて、
インライアの数や誤差の分布がどう変わるか比べてみてください。

キー操作: `o` アウトライアの表示 ON/OFF、`t` テンプレートを選び直す、`ESC` / `q` 終了

オプション:

| オプション | 説明 |
| --- | --- |
| `--camera 1` | カメラ番号（`python flann.py 1` でも可） |
| `--template ファイル名` | テンプレート画像 |
| `--feature 名前` | 局所特徴量の種類（下表、既定は `sift`） |
| `--method ransac` / `magsac` | 射影変換のロバスト推定法（既定は `ransac`） |
| `--ratio 0.7` | Lowe の ratio test の閾値（小さいほど厳しい） |
| `--reproj 5` | インライアとみなす再投影誤差の閾値 [画素]（小さいほど厳しい） |
| `--scale 0.5` | 処理時にカメラ画像を縮小する倍率 |

局所特徴量:

| `--feature` | 手法 | 記述子 | FLANN の探索法 |
| --- | --- | --- | --- |
| `sift` | SIFT (2004) | 実数 128次元 | kd-tree |
| `surf` | SURF (2006) | 実数 64次元 | kd-tree |
| `kaze` | KAZE (2012) | 実数 64次元 | kd-tree |
| `akaze` | AKAZE (2013) | バイナリ | LSH |
| `brisk` | BRISK (2011) | バイナリ | LSH |
| `orb` | ORB (2011) | バイナリ | LSH |

> **SURF について**: SURF は特許の関係で、pip で配布されている OpenCV ではビルド時に無効化されているため、通常は使えません（`--feature surf` を指定するとその旨が表示されます）。
> 同じ考え方の手法として、SIFT（特許切れのため標準搭載）や KAZE を試してください。

ロバスト推定法:

- `ransac`: 古典的な RANSAC。ランダムに選んだ4組の対応点から射影変換を求め、インライア数が最大のものを選びます。
- `magsac`: MAGSAC++。インライアかどうかを閾値で二分せず、残差に応じた重みで評価する、比較的新しい手法です（OpenCV 4.5 以降）。

### mediapipe_demo.py — MediaPipe による顔・表情・骨格・手の推定（任意）

```
python mediapipe_demo.py          # カメラ 0
python mediapipe_demo.py 1        # カメラ 1
```

Google の [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) ライブラリを使って、次の推定を行います。
それぞれキーで ON/OFF でき、同時に使うこともできます。

| キー | 内容 |
| --- | --- |
| `f` | **Face Mesh**: 顔の 478 点のメッシュと、顔の向き（鼻先から顔の右＝赤・上＝緑・正面＝青の矢印、yaw/pitch/roll の角度） |
| `b` | **表情（Blendshapes）**: まばたき・口の開き・眉の上げ下げなど 52 種類の表情の強さのうち、大きいものを棒グラフで表示 |
| `p` | **骨格（Pose）**: 全身 33 点。capture.py の `K` キーと同じ描き方なので比べてみてください |
| `h` | **手（Hand）**: 両手それぞれ 21 点と左右の判定 |
| `m` | 左右反転（鏡のように表示。最初は ON） |
| `q` / `ESC` | 終了 |

capture.py の `K` キー（`pose_estimation.py`）は、MediaPipe の骨格推定のモデルを OpenCV で動かして**中身を見せる**ためのもの、
こちらは MediaPipe ライブラリ**そのもの**を使うものです。

mediapipe をあとから入れる場合は、仮想環境を有効にしてから次を実行してください（Intel 製 CPU の Mac には対応していません）。

```
pip install -r requirements-mediapipe.txt
python download_models.py
```

## うまく動かないとき

- **カメラ画像が取れない / 真っ黒**
  - `python capture.py 1` のようにカメラ番号を変えてみる
  - Zoom や Teams など、カメラを使っている他のアプリを終了する
  - **macOS**: 初回はカメラへのアクセス許可を求められます。許可しなかった場合は「システム設定 → プライバシーとセキュリティ → カメラ」で、使っているターミナル（ターミナル.app、VS Code など）を ON にして、ターミナルを再起動してください
  - **Windows**: 「設定 → プライバシーとセキュリティ → カメラ」で「デスクトップアプリがカメラにアクセスできるようにする」を ON にする
- **キーを押しても反応しない**: ターミナルではなく、画像ウィンドウをクリックしてから押してください。日本語入力（IME）がオンだと効かないことがあります
- **矢印キーが効かない**: `+` / `-` キーを使ってください
- **`ModuleNotFoundError: No module named 'cv2'`**、**`cv2.imshow` で "The function is not implemented" エラー**、**「AKAZE / CascadeClassifier が使えません」と表示される**:
  仮想環境が有効になっていない可能性があります。プロンプトに `(.venv)` が表示されているか確認してください（「3. 仮想環境を有効にする」）。
  それでも直らなければ、`sh setup.sh --clean`（Windows は `setup.bat --clean`）で仮想環境を作り直してください
- **`python3 -m venv` が失敗する（Ubuntu など）**: `sudo apt install python3-venv` を実行してから再度セットアップしてください
- **Windows で Python を入れたのに見つからない**: Python のインストーラーで「Add python.exe to PATH」にチェックを入れてインストールし直してください
- **`pip` が見つからない**: `python -m pip ...` の形で実行してください
- **動作が重い**: `capture.py` 先頭の `PROC_SCALE` や `flann.py` の `--scale` を小さくすると速くなります

## 終了したら・削除するとき

仮想環境から抜けるには `deactivate` を実行します。
不要になったら、このフォルダごと削除すれば、インストールしたモジュールもすべて消えます（パソコン全体には何も残りません）。

## ライセンス

このリポジトリのプログラムは [MIT License](LICENSE) で公開しています。
ただし、`models/` の学習済みモデルはそれぞれのライセンスに従います（下記）。
また、研究室のロゴ（`images/interaction_lab_logo.png`）は MIT License の対象外です。

## 同梱している学習済みモデルについて

| ファイル | 出典 | ライセンス |
| --- | --- | --- |
| `models/haarcascades/*.xml` | [OpenCV](https://github.com/opencv/opencv/tree/4.x/data/haarcascades) | Intel License Agreement（各ファイル先頭に記載） |
| `models/face_detection_yunet_2023mar.onnx` | [OpenCV Zoo / YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT License（`models/LICENSE_YuNet`） |
| `models/person_detection_mediapipe_2023mar.onnx`<br>`models/pose_estimation_mediapipe_2023mar.onnx`<br>（`download_models.py` でダウンロード） | [OpenCV Zoo / MediaPipe Pose](https://github.com/opencv/opencv_zoo/tree/main/models/pose_estimation_mediapipe) | Apache License 2.0（`models/LICENSE_MediaPipe_Pose`） |
| `models/face_landmarker.task`<br>`models/pose_landmarker_full.task`<br>`models/hand_landmarker.task`<br>（`download_models.py` でダウンロード） | [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) | 各モデルのモデルカードを参照 |

Haar-like カスケードのファイルは OpenCV 4.x の pip 版には同梱されていましたが、OpenCV 5 の pip 版には含まれなくなったため、このリポジトリに同梱しています。
`pose_estimation.py` の骨格推定の前処理・後処理は OpenCV Zoo の `mp_persondet.py` / `mp_pose.py`（Apache License 2.0）を参考に書き直したもので、
ヒートマップによる補正と平滑化は MediaPipe（Apache License 2.0）の `refine_landmarks_from_heatmap_calculator.cc` と `pose_landmark_filtering.pbtxt` に合わせています。
