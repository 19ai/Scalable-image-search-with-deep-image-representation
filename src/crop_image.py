import cv2
import numpy as np
import os
import math


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

def rank(square, width, height):
    formatted = np.array([[s] for s in square], np.int32)
    x, y, wid, hei = cv2.boundingRect(formatted)
    max_distance_from_center = math.sqrt(((width / 2)) ** 2 + ((height / 2)) ** 2)
    distance_from_center = math.sqrt(((x + wid / 2) - (width / 2)) ** 2 + ((y + hei / 2) - (height / 2)) ** 2)

    height_above_horizontal = (height / 2) - y if y + hei > height / 2 else hei
    width_left_vertical = (width / 2) - x if x + wid > width / 2 else wid
    horizontal_score = abs(float(height_above_horizontal) / hei - 0.5) * 2
    vertical_score = abs(float(width_left_vertical) / wid - 0.5) * 2

    if cv2.contourArea(formatted) / (width * height) > 0.98:
        return 5  # max rank possible otherwise - penalize boxes that are the whole image heavily
    else:
        bounding_box = np.array([[[x, y]], [[x, y + hei]], [[x + wid, y + hei]], [[x + wid, y]]], dtype=np.int32)
        # every separate line in this addition has a max of 1
        return (distance_from_center / max_distance_from_center +
                cv2.contourArea(formatted) / cv2.contourArea(bounding_box) +
                cv2.contourArea(formatted) / (width * height) +
                horizontal_score +
                vertical_score)


def auto_crop(folder, file_name):
    img = cv2.imread(folder + os.sep + file_name, )
    img_copy = img.copy()[:, :, ::-1]
    height = img.shape[0]
    width = img.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # matrix of ones
    squares = []
    all_contours = []

    for gray in cv2.split(img):
        dilated = cv2.dilate(src=gray, kernel=kernel, anchor=(-1, -1))
        blured = cv2.medianBlur(dilated, 7)
        small = cv2.pyrDown(blured, dstsize=(width / 2, height / 2))
        oversized = cv2.pyrUp(small, dstsize=(width, height))
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                edges = cv2.Canny(oversized, threshold1=0, threshold2=50, apertureSize=3)
                next = cv2.dilate(src=edges, kernel=kernel, anchor=(-1, -1))
            else:
                retval, next = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            _, contours, hierarchy = cv2.findContours(next, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                all_contours.append(cnt)
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)

    sorted_squares = sorted(squares, key=lambda square: rank(square, width, height))

    img_with_top_square = img_copy.copy()
    if sorted_squares:
        print sorted_squares[0]
        cv2.drawContours(img_with_top_square, [sorted_squares[0]], -1, (0, 255, 60), 3)
        min_x, min_y = np.min(sorted_squares[0], axis=0).astype(np.int32)
        max_x, max_y = np.max(sorted_squares[0], axis=0).astype(np.int32)
        if (max_x - min_x) * (max_y - min_y) < 0.2 * (width * height):
            cv2.imwrite(folder + os.sep + 'auto_crop_' + file_name, img)
            return width, height
        else:
            cv2.imwrite(folder + os.sep + 'auto_crop_' + file_name,
                    img[min_y:max_y, min_x:max_x, :])
            return max_x -min_x, max_y - min_y

    else:
        print "There is no frame found !"
        cv2.imwrite(folder + os.sep + 'auto_crop_' + file_name,
                    img)
        return width, height


if __name__ == '__main__':
    auto_crop(folder='data1', file_name='1.jpg')
