import numpy as np
# from scipy.io import wavfile
# import librosa
# import soundfile as sf
import struct
import math
import cv2
import matplotlib.pyplot as plt
from classes.processing import Processing


# import wave


class In_Out:

    def read_dat(self, file_name):
        data = np.fromfile('data/dat/' + file_name, dtype="float32")
        return data

    """
    def read_wav(self, file_name, rate):
        out_data = dict()
        # x, _ = librosa.load('data/wav/' + file_name, sr=rate)
        # sf.write('data/wav/tmp.wav', x, rate)
        samplerate, data = wavfile.read('data/wav/' + file_name)
        out_data['rate'] = samplerate
        out_data['data'] = data
        out_data['N'] = len(data)
        return out_data

    def write_wav(self, file_name, data, rate):
        # wavfile.write('data/wav/' + file_name + '.wav', 16000, np.array(data, dtype=np.float32))
        sf.write('data/wav/' + file_name + '.wav', data, rate)
    """

    def read_jpg(self, file_name):
        img = cv2.imread('data/jpg/' + file_name + '.jpg')
        return img[:,:,0]

    def show_jpg(self, img, if_color, name):
        # fig, ax = plt.subplots(figsize=plt.figaspect(img))
        # fig.subplots_adjust(0, 0, 1, 1)

        if if_color:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='gray')
        plt.title(name, fontsize=14)
        plt.axis('off')
        plt.autoscale(tight=True)
        plt.show()

    def write_jpg(self, array, file_name):
        cv2.imwrite('data/jpg/' + file_name + '.jpg', array)

    def show_jpg_sub(self, img, name):
        # fig, ax = plt.subplots(figsize=plt.figaspect(img))
        # fig.subplots_adjust(0, 0, 1, 1)

        plt.imshow(img, cmap='gray')
        plt.title(name, fontsize=14)
        plt.axis('off')
        plt.autoscale(tight=True)

    def read_xcr(self, file_name, shape):
        new_processing = Processing()
        with open(file_name, "rb") as f:
            arr = np.fromfile(file_name, dtype='uint16')

        new_arr = arr[1024:np.size(arr) - 4096]

        img = np.zeros(int(np.size(new_arr)))
        j = 0
        for i in range(0, int(np.size(new_arr)) - 1):
            # img[i] = new_arr[j] + new_arr[j + 1] * 256
            img[i] = new_arr[i]
            j += 2
        new_img = new_processing.recount_2d(np.reshape(img, shape), 255)
        return new_img

    def show_xray(self, img, name):
        # fig, ax = plt.subplots(figsize=plt.figaspect(img))
        # fig.subplots_adjust(0, 0, 1, 1)

        plt.imshow(img, cmap='gray')
        plt.title(name, fontsize=14)
        plt.axis('off')
        plt.autoscale(tight=True)

    def reshape_bilinear_interpolation(self, image, coef):
        """
        `image` is a 2-D numpy array
        `height` and `width` are the desired spatial dimension of the new 2-D array.
        """
        img_height, img_width = image.shape[:2]
        height =  int(img_height * coef)
        width =  int(img_width * coef)
        resized = np.empty([height, width], dtype=np.uint8)

        x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
        y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

        for i in range(height-1):
            for j in range(width-1):
                x_l, y_l = math.floor(x_ratio * j), math.floor(y_ratio * i)
                x_h, y_h = math.ceil(x_ratio * j), math.ceil(y_ratio * i)

                x_weight = (x_ratio * j) - x_l
                y_weight = (y_ratio * i) - y_l

                a = image[y_l, x_l]
                b = image[y_l, x_h]
                c = image[y_h, x_l]
                d = image[y_h, x_h]

                pixel = a * (1 - x_weight) * (1 - y_weight) \
                        + b * x_weight * (1 - y_weight) + \
                        c * y_weight * (1 - x_weight) + \
                        d * x_weight * y_weight

                resized[i][j] = int(pixel)

        return resized.astype(int)

    def reshape_nearest_neighbor(self, img, coef):
        width, height = img.shape[:2]

        w1 = width
        h1 = height

        w2 = int(w1 * coef)
        h2 = int(h1 * coef)
        img_nn = np.empty((w2, h2), dtype=np.uint8)

        x_ratio = float(w1 / float(w2))
        y_ratio = float(h1 / float(h2))

        for i in range(0, w2-1):
            for j in range(0, h2-1):
                p_x = math.floor(i * x_ratio)
                p_y = math.floor(j * y_ratio)
                a = img[int(p_x), int(p_y)]
                img_nn[i, j] = a

        return img_nn