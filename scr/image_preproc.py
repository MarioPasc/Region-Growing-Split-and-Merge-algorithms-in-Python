import cv2
import numpy as np


class Preprocess:
    def __init__(self, image):
        """
        Inicializamos la clase pasando una imagen
        :param image: Imagen (ndarray 2D)
        """
        self.image = image
        self.gaussian = None

    def gaussian_filter(self, kernel_size=3, sigma=1):
        """
        Aplicamos una reducción de ruido a la imagen con un filtro
        Gaussiano

        :param sigma: Float, representa la desviación estandar para el kernel Gaussiano
        :param kernel_size: Int, representa el tamaño del kernel en una dirección (5 para 5x5, p.ej.)
        :return: Devuelve la imagen con la reducción de ruido
        """
        # Centrando el kernel en 0, sacamos cuántas unidades tiene simétricamente hacia cada eje
        offset = int(kernel_size // 2)
        # De la malla generada para X, y en el rango -offset:offset+1, se separa en la malla del eje X y del eje y
        x, y = np.mgrid[-offset:offset + 1, -offset:offset + 1]
        # Teniendo estos valores para rellenar, se aplica la función gaussiana
        self.gaussian = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    def convolve(self, kernel):
        """
        Aplica la convolución definida en la librería OpenCV entre el kernel
        y los datos del objeto

        :param kernel: 2D numpy array
        :return: Imagen con el filtro aplicado, tiene la misma dimensión que la imagen de entrada (-1)
        """
        return cv2.filter2D(self.image, -1, kernel)

    def normalize(self, dtype=None):
        """
        Normaliza la imagen al rango especificado
        :param dtype: Tipo de dato al que se quiere normalizar, si es None, es al rango [0, 1]
        :return:
        """
        max = np.max(self.image)
        min = np.min(self.image)
        self.image = (self.image - min)/(max - min)
        if dtype is not None:
            max_val = np.iinfo(dtype).max
            self.image = (self.image * max_val).astype(dtype)


#%%
