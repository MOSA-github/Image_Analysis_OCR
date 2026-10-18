import cv2
import numpy as np
import math

def remove_lines(img):
    # 1. グレースケール化
    if len(img.shape) == 2:
        gray = img.copy()
    elif len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"対応していない画像shapeです: {img.shape}")

    # ─── 改善ポイント①: 画像を拡大して線をはっきりさせる ───
    # 低解像度のままだと線が途切れるため、2倍に拡大します
    h, w = gray.shape[:2]
    gray_resized = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # ─── 改善ポイント②: エッジ検出の前にコントラストを上げ、ボケを抑える ───
    # 微小な格子線を強調するために、適応的ヒストグラム平坦化（CLAHE）を適用
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray_resized)

    # 2. エッジ検出
    # 線が細く薄い場合があるため、閾値を少し下げて感度を上げます
    edges = cv2.Canny(gray_enhanced, 30, 100)
    cv2.imwrite("edges_debug.png", edges) # ここで格子線が白く浮き出ているか確認してください

    # 3. Hough変換
    # ─── 改善ポイント③: パラメータのチューニング ───
    # 途切れた線をつなぎやすくするため maxLineGap を広げ、短い線も拾うため minLineLength を下げます
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,       # 50 -> 30 (検出基準を下げる)
        minLineLength=15,   # 30 -> 15 (短い途切れ線も拾う)
        maxLineGap=20       # 10 -> 20 (離れた線同士を繋ぐ)
    )

    mask = np.zeros_like(gray_enhanced)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line # HoughLinesPの結果は通常二重リストです

            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if angle < 0:
                angle += 180

            # ─── 改善ポイント④: 角度の許容誤差を広げる ───
            # カメラの傾きを考慮して、±5° から ±8° に広げます
            is_45 = 37 <= angle <= 53
            is_135 = 127 <= angle <= 143

            if is_45 or is_135:
                # 線の描画（少し太めにマスクを作るとインペイントで消えやすいです）
                cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=4) # 3 -> 4

    cv2.imwrite("mask_debug.png", mask) # ここで格子線の上に綺麗に白い線が乗っているか確認

    # 4. 線を除去
    result_resized = gray_enhanced.copy()
    result_resized[mask == 255] = 255

    # 元のサイズに戻す（PaddleOCRに渡すサイズに合わせる）
    result = cv2.resize(result_resized, (w, h), interpolation=cv2.INTER_AREA)

    return result
