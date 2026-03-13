# import
import cv2
import numpy as np
from itertools import islice
from piece_classifier import PieceClassifier

classifier = PieceClassifier()

def get_coords(img_board, img_board_gray, pics) -> list[tuple]:
    coords = []
    for src in pics:
        img_piece = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        mask = img_piece[:, :, 3]  # use the inverted transparency channel for mask

        # Convert both images to grayscale
        img_piece_gray = cv2.cvtColor(img_piece, cv2.COLOR_BGR2GRAY)
        h, w = img_piece_gray.shape

        result = cv2.matchTemplate(img_board_gray, img_piece_gray, cv2.TM_SQDIFF_NORMED, mask=mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        top_left = min_loc
        bottom_right = (top_left[0] + img_piece.shape[1], top_left[1] + img_piece.shape[0])

        # overwrite the portion of the result that has the match:
        h1 = top_left[1] - h // 2
        h1 = np.clip(h1, 0, result.shape[0])

        h2 = top_left[1] + h // 2 + 1
        h2 = np.clip(h2, 0, result.shape[0])

        w1 = top_left[0] - w // 2
        w1 = np.clip(w1, 0, result.shape[1])

        w2 = top_left[0] + w // 2 + 1
        w2 = np.clip(w2, 0, result.shape[1])
        result[h1:h2, w1:w2] = 1  # poison the result in the vicinity of this match so it isn't found again

        # look for next match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        coords.append(top_left)
        coords.append((h, w))

    top_left = coords[2]
    x_coord = top_left[0]
    x_coord += int(coords[3][0] * 1.35)
    y_coord = top_left[1]
    y_coord -= int(coords[3][1] * 3)
    top_left = (x_coord, y_coord)

    # Fix bottom right coord
    bottom_right = coords[0]
    x_coord = bottom_right[0]
    x_coord += int(coords[1][0] * 2.95)
    y_coord = bottom_right[1]
    y_coord -= int(coords[1][1] * 0.65)

    bottom_right = (x_coord, y_coord)
    cv2.rectangle(img_board, top_left, bottom_right, (0, 255, 255), 2)
    return [top_left, bottom_right]


def get_pieces(img_board, board_minx, board_miny):

    coords = []

    square_size = 95

    for row in range(8):
        for col in range(8):

            x = board_minx + col*square_size
            y = board_miny + row*square_size

            square = img_board[y:y+square_size, x:x+square_size]

            piece = classifier.predict(square)

            if piece != "":
                coords.append(piece)
                coords.append((x,y))

    return coords
