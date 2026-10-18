# Image_Analysis_OCR
院内の見える化システムの画像解析
paddleOCRを用いた画像からの文字読み取り

## Install paddleOCR
paddleOCRのモデルをインストールする必要があります．
インストール手順については以下のページを参照．
https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html#21-command-line

## Usage
paddleOCR.py内の変数video_pathに解析対象とするMP4ファイルのパスを設定．
paddleOCR.pyを実行．
各フレームの解析結果がresultsフォルダに.jsonファイルと.pngファイルとして保存される．
