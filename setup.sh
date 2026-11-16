#!/bin/sh
# ----------------------------------------------------------------------
# セットアップ用スクリプト（macOS / Linux）
#
#   sh setup.sh           仮想環境 .venv を作り、必要なモジュールを入れる
#   sh setup.sh --clean   .venv を削除して作り直す（うまく動かないとき）
#
# パソコン全体の Python には何もインストールしません。
# すべてこのフォルダの .venv の中に入ります。
# ----------------------------------------------------------------------
set -e
cd "$(dirname "$0")"

# --- Python 3.8 以上を探す ---
PY=""
for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1 &&
       "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1; then
        PY="$cand"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "Python 3.8 以上が見つかりません。Python をインストールしてから再実行してください。"
    exit 1
fi
echo "使用する Python: $(command -v "$PY") ($("$PY" --version 2>&1))"

if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "（注意）Anaconda の環境「$CONDA_DEFAULT_ENV」が有効になっています。"
    echo "        このまま続けても構いませんが、うまくいかないときは"
    echo "        conda deactivate を実行してから再度試してください。"
fi

# --- 仮想環境の作成 ---
if [ "$1" = "--clean" ] && [ -d .venv ]; then
    echo ".venv を削除して作り直します..."
    rm -rf .venv
fi

if [ ! -x .venv/bin/python ]; then
    echo "仮想環境 .venv を作成しています..."
    if ! "$PY" -m venv .venv; then
        echo ""
        echo "仮想環境を作成できませんでした。"
        echo "Ubuntu などでは  sudo apt install python3-venv  が必要な場合があります。"
        rm -rf .venv
        exit 1
    fi
else
    echo "既存の仮想環境 .venv を使います（作り直す場合: sh setup.sh --clean）"
fi

# --- モジュールのインストール（.venv の Python を直接使う）---
echo "必要なモジュールをインストールしています（数分かかることがあります）..."
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# --- 環境チェック ---
echo ""
.venv/bin/python check_env.py || true

cat <<'EOF'

----------------------------------------------------------------------
セットアップが終わりました。
ターミナルを開くたびに、まず次のコマンドで仮想環境を有効にしてください。

    source .venv/bin/activate

プロンプトの先頭に (.venv) と表示されたら、次のように実行できます。

    python capture.py
    python flann.py
----------------------------------------------------------------------
EOF
