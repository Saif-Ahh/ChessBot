import tensorflow as tf
import cv2
import numpy as np

IMG_SIZE = 96

CLASS_NAMES = [
    "empty",
    "white_pawn","white_knight","white_bishop","white_rook","white_queen","white_king",
    "black_pawn","black_knight","black_bishop","black_rook","black_queen","black_king"
]

PIECE_MAP = {
    "white_pawn":"P",
    "white_knight":"N",
    "white_bishop":"B",
    "white_rook":"R",
    "white_queen":"Q",
    "white_king":"K",
    "black_pawn":"p",
    "black_knight":"n",
    "black_bishop":"b",
    "black_rook":"r",
    "black_queen":"q",
    "black_king":"k"
}


class PieceClassifier:

    def __init__(self, model_path="chess_piece_classifier.keras"):
        self.model = tf.keras.models.load_model(model_path)

    def preprocess(self, img):

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype("float32")/255.0

        return np.expand_dims(img,0)

    def predict(self,img):

        img = self.preprocess(img)

        probs = self.model.predict(img,verbose=0)[0]
        idx = np.argmax(probs)

        label = CLASS_NAMES[idx]

        if label == "empty":
            return ""

        return PIECE_MAP[label]