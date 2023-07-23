import cv2
import numpy as np


def cv_put_text(img_arr: np.ndarray,
                text,
                coord,
                color,
                scale=1.0,
                width=2,
                border=False,
                border_width=1.5,
                border_color=(0, 0, 0)):
    if border:
        cv2.putText(img_arr, text, coord, cv2.FONT_HERSHEY_SIMPLEX, scale, border_color,
                    int(width + border_width), cv2.LINE_AA)
    cv2.putText(img_arr, text, coord, cv2.FONT_HERSHEY_SIMPLEX, scale, color, width, cv2.LINE_AA)


def cv_rectangle(img_arr: np.ndarray,
                 xy_start,
                 xy_end,
                 color,
                 width,
                 border=False,
                 border_width=1.5,
                 border_color=(0, 0, 0)):
    if border:
        cv2.rectangle(img_arr, xy_start, xy_end, border_color, int(width + border_width))
    cv2.rectangle(img_arr, xy_start, xy_end, color, width)