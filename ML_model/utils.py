import tensorflow as tf
import cv2
import numpy as np
class MNIST:
    model = None
    __instance = None
    @staticmethod
    def getInstance():
        """ Static access method. """
        if MNIST.__instance == None:
            MNIST()
        return MNIST.__instance
    def predict(self,img):
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = cv2.bitwise_not(img2)
        img2 = img2 / 255.0
        img2 = cv2.resize(img2, (28, 28))
        img2 = np.reshape(img2, (1, 28, 28, 1))
        result = MNIST.__instance.model.predict(img2)
        return np.argmax(result)
    def __init__(self):
        if MNIST.__instance ==None:
            MNIST.__instance = self
            MNIST.__instance.model = tf.keras.models.load_model('ML_model/kaggle_MNIST_1.h5')

class Julia:
    model = None
    __instance = None
    num_dict = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
                'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e',
                'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    @staticmethod
    def getInstance():
        """ Static access method. """
        if Julia.__instance == None:
            Julia()
        return Julia.__instance
    def predict(self,img):
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = cv2.bitwise_not(img2)
        img2 = img2 / 255.0
        img2 = cv2.resize(img2, (32, 32))
        img2 = np.reshape(img2, (1, 32, 32, 1))
        result = Julia.__instance.model.predict(img2)
        return self.num_dict[int(np.argmax(result))]
    def __init__(self):
        if Julia.__instance ==None:
            Julia.__instance = self
            Julia.__instance.model = tf.keras.models.load_model('ML_model/kaggle_First_Steps_With_Julia_1.h5')