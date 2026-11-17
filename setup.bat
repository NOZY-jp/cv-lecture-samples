@echo off
rem ----------------------------------------------------------------------
rem Setup script for Windows
rem
rem   setup.bat           create .venv and install required modules
rem   setup.bat --clean   delete .venv and recreate it
rem
rem You can also run it by double-clicking in Explorer.
rem ----------------------------------------------------------------------
setlocal
chcp 65001 >nul
cd /d "%~dp0"

rem --- Python 3.8 以上を探す（py ランチャー → python の順）---
set "PY="
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo Python 3.8 以上が見つかりません。
    echo https://www.python.org/downloads/ からインストールしてください。
    echo インストール時に "Add python.exe to PATH" にチェックを入れてください。
    goto :fail
)
echo 使用する Python:
%PY% -c "import sys; print(sys.executable, sys.version.split()[0])"

if defined CONDA_DEFAULT_ENV (
    echo （注意）Anaconda の環境「%CONDA_DEFAULT_ENV%」が有効になっています。
    echo         うまくいかないときは conda deactivate を実行してから再度試してください。
)

rem --- 仮想環境の作成 ---
if /i "%~1"=="--clean" if exist .venv (
    echo .venv を削除して作り直します...
    rmdir /s /q .venv
)

if not exist .venv\Scripts\python.exe (
    echo 仮想環境 .venv を作成しています...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo 仮想環境を作成できませんでした。
        goto :fail
    )
) else (
    echo 既存の仮想環境 .venv を使います（作り直す場合: setup.bat --clean）
)

rem --- モジュールのインストール（.venv の Python を直接使う）---
echo 必要なモジュールをインストールしています（数分かかることがあります）...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :fail

rem --- mediapipe（任意。mediapipe_demo.py 用。入らない環境でも続行）---
echo.
echo mediapipe（任意）をインストールしています...
.venv\Scripts\python.exe -m pip install -r requirements-mediapipe.txt
if errorlevel 1 echo （mediapipe はこの環境にはインストールできませんでした。mediapipe_demo.py 以外は使えます）

rem --- 学習済みモデルのダウンロード（失敗しても続行）---
echo.
echo 学習済みモデルをダウンロードしています...
.venv\Scripts\python.exe download_models.py

rem --- 環境チェック ---
echo.
.venv\Scripts\python.exe check_env.py

echo.
echo ----------------------------------------------------------------------
echo セットアップが終わりました。
echo コマンドプロンプトを開くたびに、まず次のコマンドで仮想環境を有効にしてください。
echo.
echo     .venv\Scripts\activate
echo.
echo プロンプトの先頭に (.venv) と表示されたら、次のように実行できます。
echo.
echo     python capture.py
echo     python flann.py
echo     python mediapipe_demo.py   （mediapipe が入った場合）
echo ----------------------------------------------------------------------
pause
exit /b 0

:fail
echo.
echo セットアップに失敗しました。上のメッセージを確認してください。
pause
exit /b 1
