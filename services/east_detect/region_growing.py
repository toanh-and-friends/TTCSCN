import os
import cv2
import numpy as np


def region_growing(img, x, y, delta=15):
    # img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
    """
        x: relative coord (0-1)
        y: relative coord (0-1)
    """

    def is_valid_px(x, y, img):
        if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0]:
            return True
        return False

    mask = np.zeros(img.shape[:2])
    y_abs = int(y * img.shape[0])
    x_abs = int(x * img.shape[1])

    loop_points = []
    loop_points.append((x_abs, y_abs))
    px_intensity = img[y_abs, x_abs, :]
    blue = px_intensity[0]
    green = px_intensity[1]
    red = px_intensity[2]

    while len(loop_points) != 0:
        (cx, cy) = loop_points.pop()
        current_intensity = img[cy, cx, :]
        b = int(current_intensity[0])
        g = int(current_intensity[1])
        r = int(current_intensity[2])

        if abs(r - red) < delta and abs(g - green) < delta and abs(b - blue) < delta:
            mask[cy, cx] = 255
            neighbors = [
                (cx - 1, cy),
                (cx + 1, cy),
                (cx - 1, cy - 1),
                (cx, cy - 1),
                (cx + 1, cy - 1),
                (cx - 1, cy + 1),
                (cx, cy + 1),
                (cx + 1, cy + 1)
            ]
            for (nx, ny) in neighbors:
                if is_valid_px(nx, ny, img) and mask[ny, nx] == 0:
                    loop_points.append((nx, ny))
        else:
            pass
    return mask


def main(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    KERNEL_WIDTH = KERNEL_HEIGHT = 5
    SIGMA_X = SIGMA_Y = 2
    img[:, :, 0] = cv2.GaussianBlur(img[:, :, 0], ksize=(KERNEL_WIDTH, KERNEL_HEIGHT), sigmaX=SIGMA_X, sigmaY=SIGMA_Y)
    img[:, :, 1] = cv2.GaussianBlur(img[:, :, 1], ksize=(KERNEL_WIDTH, KERNEL_HEIGHT), sigmaX=SIGMA_X, sigmaY=SIGMA_Y)
    img[:, :, 2] = cv2.GaussianBlur(img[:, :, 2], ksize=(KERNEL_WIDTH, KERNEL_HEIGHT), sigmaX=SIGMA_X, sigmaY=SIGMA_Y)
    cv2.imwrite('out_blur_' + os.path.basename(img_path), img)

    x = 0.5
    y = 0.6
    delta = 20
    mask = region_growing(img=img, x=x, y=y, delta=delta)
    cv2.imwrite('out_mask.jpg', mask)

    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_color[mask > 0, :] = (0, 255, 255)

    cv2.circle(img_color, (int(x * img.shape[1]), int(y * img.shape[0])), radius=5, color=(0, 0, 255), thickness=2)
    cv2.imwrite('out_seg_' + os.path.basename(img_path), img_color)


if __name__ == "__main__":
    print('Running...')
    main('../datasets/ICDAR2015/test_data/stop.jpeg')
