import cv2
import matplotlib.pyplot as plt

def save_histogram(img, filename):
    # グレースケール画像ならそのまま
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ヒストグラム計算
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # 描画
    plt.figure()
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")

    # 保存
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()