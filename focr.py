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
    w = x1-x0
    h = y1-y0
    if w < 5 or h < 5:
        return ''
    size_max = max(w, h)
    letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
    if w > h:
        y_pos = size_max//2 - h//2
        letter_square[y_pos:y_pos + h, 0:w] = letter_crop
    elif w < h:
        x_pos = size_max//2 - w//2
        letter_square[0:h, x_pos:x_pos + w] = letter_crop
    else:
        letter_square = letter_crop
    letter = (x, w, cv.resize(letter_square, (28, 28), interpolation=cv.INTER_AREA))
    im2arr = cv.bitwise_not(letter[2])
    im2arr = np.reshape(im2arr,(1,784))
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
    img1 = img&224
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
    kernel = np.ones((3,3),np.uint8)
    mask = cv.dilate(vis,kernel,iterations = 1)
    return ~mask

if __name__ == '__main__':
    if len (sys.argv) > 1 :
        in_file = sys.argv[1]
    else:
        if os.path.exists('turbobit_net_v50_GDL_03.png'):
            in_file = 'turbobit_net_v50_GDL_03.png'
        else:
            sys.exit()
    if len (sys.argv) > 2 :
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
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    blur = cv.blur(resized,(3,3))
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    img0 = gray.copy()
    thresh = cv.inRange(img0, 170, 246)
    cv.line(thresh, (0, 4), (300, 4), (255), 10)
    cv.line(thresh, (0, 96), (300, 96), (255), 10)
    imask = get_mask(resized)
    thresh = cv.bitwise_or(thresh, imask)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img1 = img0.copy()
    img1.fill(255)
    cv.fillPoly(img1, contours, 0)
    mask = cv.bitwise_not(img1)
    rects = []
    nn = NeuralNetMLP()
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv.boundingRect(contour)
        if hierarchy[0][idx][3] == 0 and w > 5 and h > 10:
            cv.rectangle(thresh, (x, y), (x + w, y + h), (120, 0, 0), 1)
            rects.append([x, y, w, h])
    if len(rects) == 4:
        for x, y, w, h in sorted(rects, key=operator.itemgetter(0)):
            cv.rectangle(thresh, (x, y), (x + w, y + h), (10, 0, 0), 1)
            symb = get_symb(mask, x, x+w, y,y+h)
            rez += symb
    else:
        blur = cv.blur(resized,(5,5))#cv.GaussianBlur(resized,(5,5),0)#
        gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        thresh = cv.inRange(img0, 170, 246)#cv.threshold(img0,150,255,cv.THRESH_BINARY+cv.THRESH_OTSU)#
        thresh = cv.bitwise_or(thresh, imask)
        thresh = restore_lines(thresh)
        cv.line(thresh, (0, 4), (300, 4), (255), 10)
        cv.line(thresh, (0, 96), (300, 96), (255), 10)
        for i, slick in enumerate(parts):
            edge0, edge1 = slick
            img0 = thresh[0:100, 2*edge0:2*edge1]
            img1 = img0.copy()
            cv.line(img1, (0, 0), (0, 100), (255), 4)
            cv.line(img1, (2*(edge1-edge0-1), 0), (2*(edge1-edge0-1), 100), (255), 2)
            contours, hierarchy = cv.findContours(img1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            img1.fill(255)
            cv.fillPoly(img1, contours, 0)
            mask = cv.bitwise_not(img1)
            x0, y0 = 255,255
            x1, y1 = 0, 0
            cnts = []
            for idx, contour in enumerate(contours):
                (x, y, w, h) = cv.boundingRect(contour)
                if hierarchy[0][idx][3] == 0 and w > 7 and h > 7:
                            cnts.append([x, y, w, h, w*h])
            for cn in sorted(cnts, key=operator.itemgetter(4), reverse = True):
                (x, y, w, h, _) = cn
                if x1 == 0 or x-x1 < 6:
                    if x < x0:
                        x0 = x
                    if x+w > x1:
                        x1 = x+w
                if y1 == 0 or y-y1 < 4:
                    if y < y0:
                        y0 = y
                    if y+h > y1:
                        y1 = y+h
            cv.rectangle(img0, (x0, y0), (x1, y1), (10, 0, 0), 1)
            symb = get_symb(mask, x0, x1, y0,y1)
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