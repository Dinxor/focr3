import os
import sys
import time
import pyperclip
import configparser
import operator
from PIL import Image
import numpy as np
import cv2 as cv
import pickle

class NeuralNetMLP(object):

    def __init__(self, filename = 'turbobit.sav'):
        self.symbs = ['2','3','4','5','6','7','9','A','C','D','E','F','H','J','K','L','M','N','P','R','S','T','U','V','W','X','Y','Z']
        file = open(filename, 'rb')
        param=pickle.load(file)
        [self.w_h, self.w_out, self.b_h, self.b_out] = param

    def _sigmoid(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        z_h = np.dot(X, self.w_h) + self.b_h
        a_h = self._sigmoid(z_h)
        z_out = np.dot(a_h, self.w_out) + self.b_out
        a_out = self._sigmoid(z_out)
        return z_h, a_h, z_out, a_out

    def predict(self, X):
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return self.symbs[y_pred[0]] if len(y_pred) == 1 else ''

def get_symb(img, x0, x1, y0, y1):
    letter_crop = img[y0:y1, x0:x1]
    w = x1 - x0
    h = y1 - y0
    if w < 5 or h < 5:
        return ''
    size_max = max(w, h)
    letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
    if w > h:
        y_pos = size_max // 2 - h // 2
        letter_square[y_pos:y_pos + h, 0:w] = letter_crop
    elif w < h:
        x_pos = size_max // 2 - w // 2
        letter_square[0:h, x_pos:x_pos + w] = letter_crop
    else:
        letter_square = letter_crop
    resized_letter = cv.resize(letter_square, (28, 28), interpolation=cv.INTER_AREA)
    im2arr = cv.bitwise_not(resized_letter)
    im2arr = np.reshape(im2arr, (1, 784))
    symb = nn.predict(im2arr)
    return symb

def restore_lines(distorted):
    edges = cv.Canny(distorted, 50, 150, apertureSize=7) 
    lines = cv.HoughLinesP(image=edges, rho=0.8, theta=np.pi/360, threshold=40, lines=np.array([]), minLineLength=20, maxLineGap=2)
    if lines is not None:
        a, b, c = lines.shape
        for i in range(a):
            cv.line(distorted, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2],  lines[i][0][3]), (255), 3, cv.LINE_AA)
    return distorted

def get_mask(img):
    height, width,_ = img.shape
    stop = set()
    img1 = img & 224
    for i in range(width):
        stop.add(tuple(img1[0,i]))
        stop.add(tuple(img1[height-1,i]))
    d = {}
    vis = np.zeros(img.shape[:2], np.uint8)
    for i in range(width):
        for j in range(height):
            color = tuple(img1[j,i])
            if not color in stop:
                vis[j,i] = 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.dilate(vis, kernel, iterations=1)
    return ~mask

def find_symbol_borders(contour):
    x, y, w, h = cv.boundingRect(contour)
    return x, y, x + w, y + h

def extract_symbols_from_image(image, parts):
    symbols = []
    contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv.contourArea(contour) > 50:  # filter out too small contours
            x0, y0, x1, y1 = find_symbol_borders(contour)
            if x1 > x0 and y1 > y0:
                symbols.append((x0, y0, x1, y1))
    return symbols

if __name__ == '__main__':
    if len(sys.argv) > 1:
        in_file = sys.argv[1]
    else:
        if os.path.exists('ozo sos zoz.png'):
            in_file = 'ozo sos zoz.png'
        else:
            sys.exit()
    
    if len(sys.argv) > 2:
        out_name = sys.argv[2]
    else:
        out_name = ''
    
    path = "focr.ini"
    changes = []
    koefs = []
    if os.path.exists(path):
        config = configparser.ConfigParser()
        config.read(path)
        save_img = int(config.get("Image", "save_img"))
        save_parts = int(config.get("Image", "save_parts"))
        if save_img + save_parts > 0:
            save_dir = config.get("Image", "save_dir")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            basename = str(int(time.time()))
        mode = config.get("Output", "mode")
        if mode == 'file' and out_name == '':
            out_name = config.get("Output", "filename")
        parts = []
        for option in config.options("Parts"):
            parts.append([int(option), int(config.get("Parts", option))])
    else:
        save_img, save_parts = 0, 0
        mode = 'file'
        if out_name == '':
            out_name = 'rezocr.txt'
        parts = [[0,43], [35,75], [70,107], [102, 148]]

    rez = ''
    img = cv.imread(in_file)
    dim = (300,100)
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    blur = cv.blur(resized, (3, 3))
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    img0 = gray.copy()
    thresh = cv.inRange(img0, 170, 246)
    cv.line(thresh, (0, 4), (300, 4), (255), 10)
    cv.line(thresh, (0, 96), (300, 96), (255), 10)
    imask = get_mask(resized)
    thresh = cv.bitwise_or(thresh, imask)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img1 = img0.copy()
    img1.fill(255)
    cv.fillPoly(img1, contours, 0)
    mask = cv.bitwise_not(img1)
    nn = NeuralNetMLP()

    # Extract symbols from contours
    symbols = extract_symbols_from_image(mask, parts)
    symbols.sort(key=lambda x: x[0])  # Sort by x-coordinate to read from left to right
    for x0, y0, x1, y1 in symbols:
        symb = get_symb(mask, x0, x1, y0, y1)
        rez += symb

    if save_parts > 0:
        basename = str(int(time.time()))
        thresh.save(save_dir + basename + '_' + rez + '_.bmp')

    if mode == 'file':
        fout = open(out_name, 'w')
        fout.write(rez)
        fout.close()
    elif mode == 'clipboard':
        pyperclip.copy(rez)
    elif mode == 'print':
        print(rez)
    elif mode == 'stdout':
        sys.stdout.write(rez)
    if save_img > 0:
        os.system('copy ' + str(in_file) + ' ' + save_dir + basename + '_' + rez + '.png > nul')
    time.sleep(5)
