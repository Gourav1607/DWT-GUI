#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# DWT-Haar / Multimedia Processing
# Gourav Siddhad
# 14-October-2019


# In[ ]:


print('Importing Libraries', end='')

import cv2
import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, Canvas, PhotoImage
from PIL import Image, ImageTk

print(' - Done')


# In[20]:


def calc_error(img1, img2):
    return np.array(img1-img2, dtype='uint8')

def calc_dwt(img, level, th, thflag=False):
    maxlevel = np.log2(len(img))
    
    if (maxlevel-int(maxlevel))>0:
        maxlevel = int(maxlevel) + 1
    totval = np.power(2, maxlevel)
    
    nimg = np.zeros((int(totval)))
    nimg[0:len(img)] = img
    
    val = int(np.power(2, level))
    proc = nimg[0:val]
    done = nimg[val:]
    
    out1, out2 = [], []
    i = 0
    while(i<len(proc)):
        v = (proc[i] + proc[i+1])/2
        if thflag:
            if v>=th:
                v = v
            else:
                v = 0
        out1.append(v)
        
        v = (proc[i] - proc[i+1])/2
        if thflag:
            if v>=th:
                v = v
            else:
                v = 0
        out2.append(v)
        i += 2

    final = []
    final.extend(out1)
    final.extend(out2)
    final.extend(done)
    
    return np.array(final)


# In[21]:


def calc_idwt(img, level):
    maxlevel = np.log2(len(img))
    if (maxlevel-int(maxlevel))>0:
        maxlevel = int(maxlevel) + 1
    totval = np.power(2, maxlevel)
    nimg = np.zeros((int(totval)))
    nimg[0:len(img)] = img
    
    val = int(np.power(2, level-1))
    proc = nimg[0:val]
    proc2 = nimg[val:2*val]
    done = nimg[2*val:]
    
    out = []
    for i, n in enumerate(proc):
        out.append(proc[i] + proc2[i])
        out.append(proc[i] - proc2[i])
        
    final = []
    final.extend(out)
    final.extend(done)
    
    return np.array(final)


# In[22]:


def calc_dwt_gray(img, maxl, minl, th, thFlag=False):
    timg = np.array(img)
    for i in range(maxl, minl-1, -1):
        nimg = []
        for row in timg:
            nimg.append(calc_dwt(row, i, th, thFlag))
        timg = np.array(nimg)
        nimg = []
        for row in timg.T:
            nimg.append(calc_dwt(row, i, th, thFlag))
        timg = np.array(nimg).T
    return timg
    
def calc_idwt_gray(img, maxl, minl):
    timg = np.array(img)
    for i in range(minl, maxl+1, 1):
        nimg = []
        for row in timg:
            nimg.append(calc_idwt(row, i))
        timg = np.array(nimg)
        nimg = []
        for row in timg.T:
            nimg.append(calc_idwt(row, i))
        timg = np.array(nimg).T
    return timg


# In[23]:


class myimg():
    def __init__(self):
        self.img = None
        self.dwt = None
        self.idwt = None
        self.maxl = 8
        self.minl = 8
        self.th = 0
        self.thflag = False
        
    def set_img(self, img):
        self.img = img
    
    def set_dwt(self, dwt):
        self.dwt = dwt
    
    def set_idwt(self, idwt):
        self.idwt = idwt
    
    def set_maxl(self, maxl):
        self.maxl = maxl
        
    def set_minl(self, minl):
        self.minl = minl
        
    def set_th(self, th):
        self.th = th
        
    def set_thflag(self, thflag):
        self.thflag = thflag
        
        
    def get_img(self):
        return self.img
    
    def get_dwt(self):
        return self.dwt
        
    def get_idwt(self):
        return self.dwt
    
    def get_maxl(self):
        return self.maxl
    
    def get_minl(self):
        return self.minl
    
    def get_th(self):
        return self.th
    
    def get_thflag(self):
        return self.thflag


# In[34]:


global mimg
mimg = myimg()

master = tk.Tk()
master.title('DWT - Haar Wavelet')
width, height = 1300, 400
xcord = master.winfo_screenwidth() // 2 - width // 2
ycord = master.winfo_screenheight() // 2 - height // 2
master.geometry("%dx%d+%d+%d" % (width, height-20, xcord, ycord-20))

def load_image():
    initdir = '/Documents/'
    filename = filedialog.askopenfilename(initialdir = initdir, title = 'Select Image')

    oimg = cv2.imread(filename)
    if len(oimg.shape)==2:
        img = cv2.resize(oimg, (256, 256), cv2.INTER_AREA)
    else:
        oimg = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(oimg, (256, 256), cv2.INTER_AREA)
        
    mimg.set_img(img)
    
    render = ImageTk.PhotoImage(Image.fromarray(img))
    oimg_label0 = tk.Label(firstFrame, image=render)
    oimg_label0.image = render
    oimg_label0.grid(row = 1, column = 0)
    
# First Coumn
firstFrame = tk.Frame(master, padx=5, pady=5)
firstFrame.grid(row = 0, column = 0)
labelInput = tk.Label(firstFrame, text = 'Input Image')
labelInput.grid(row = 0, column = 0)
oimg_label0 = tk.Label(firstFrame, height=20, width=20)
oimg_label0.grid(row = 1, column = 0)
oimg = tk.Button(firstFrame, text='Open Image', command=load_image)
oimg.grid(row = 2, column = 0)

# Second Column
secondFrame = tk.Frame(master, padx=5, pady=5)
secondFrame.grid(row = 0, column = 1)
optionScalerl = tk.Scale(secondFrame, label='Resolution Level', from_=1, to=8, orient = tk.HORIZONTAL, resolution=1)
optionScalerl.set(8)
optionScalerl.grid(row = 1, column = 0)
thresflag = tk.IntVar()
optioncb = tk.Checkbutton(secondFrame, text="Threshold", variable=thresflag)
optioncb.grid(row = 2, column = 0)
optionScaleth = tk.Scale(secondFrame, label='Threshold Value', from_=-50, to=50, orient = tk.HORIZONTAL, resolution=1)
optionScaleth.set(0)
optionScaleth.grid(row = 3, column = 0)

def decompose_img():
    mimg.set_minl(int(optionScalerl.get()))
    mimg.set_th(int(optionScaleth.get()))
    if thresflag.get() is 0:
        mimg.set_thflag(False)
    else:
        mimg.set_thflag(True)

    maxl = mimg.get_maxl()
    minl = mimg.get_minl()
    th = mimg.get_th()
    thFlag = mimg.get_thflag()

    img = np.array(mimg.get_img())
    dwt_img = np.array(img)
    if len(img.shape) == 2:
        dwt_img = calc_dwt_gray(dwt_img, maxl, minl, th, thFlag)
    else:
        for i in range(3):
            dwt_img[:, :, i] = calc_dwt_gray(dwt_img[:, :, i], maxl, minl, th, thFlag)
    dwt_img = np.array(dwt_img)
    mimg.set_dwt(dwt_img)
    
    dimg = Image.fromarray(dwt_img)
    render = ImageTk.PhotoImage(dimg)
    oimg_label2 = tk.Label(thirdFrame, image=render)
    oimg_label2.image = render
    oimg_label2.grid(row = 1, column = 0)
        
def reconstruct_img():
    maxl = mimg.get_maxl()
    minl = mimg.get_minl()
    th = mimg.get_th()
    thFlag = mimg.get_thflag()
    
    dimg = mimg.get_dwt()
    idwt_img = np.array(dimg)
    if len(dimg.shape) == 2:
        idwt_img = calc_idwt_gray(idwt_img, maxl, minl)
    else:
        for i in range(3):
            idwt_img[:, :, i] = calc_idwt_gray(idwt_img[:, :, i], maxl, minl)
    idwt_img = np.array(idwt_img)
    mimg.set_idwt(idwt_img)
    
    idimg = Image.fromarray(idwt_img)
    render = ImageTk.PhotoImage(idimg)
    oimg_label3 = tk.Label(fourthFrame, image=render)
    oimg_label3.image = render
    oimg_label3.grid(row = 1, column = 0)
    
    err = calc_error(mimg.get_img(), idwt_img)
    err = Image.fromarray(err)
    erender = ImageTk.PhotoImage(err)
    oimg_label4 = tk.Label(fifthFrame, image=erender)
    oimg_label4.image = erender
    oimg_label4.grid(row = 1, column = 0)

# Third Column
thirdFrame = tk.Frame(master, padx=5, pady=5)
thirdFrame.grid(row = 0, column = 2)
labelInput = tk.Label(thirdFrame, text = 'DWT')
labelInput.grid(row = 0, column = 0)
oimg_label2 = tk.Label(thirdFrame, height=20, width=20)
oimg_label2.grid(row = 1, column = 0)
oimgd = tk.Button(thirdFrame, text='Decompose Image', command=decompose_img)
oimgd.grid(row = 2, column = 0)

# Fourth Column
fourthFrame = tk.Frame(master, padx=5, pady=5)
fourthFrame.grid(row = 0, column = 3)
labelInput = tk.Label(fourthFrame, text = 'iDWT')
labelInput.grid(row = 0, column = 0)
oimg_label3 = tk.Label(fourthFrame, height=20, width=20)
oimg_label3.grid(row = 1, column = 0)
oimgr = tk.Button(fourthFrame, text='Reconstruct Image', command=reconstruct_img)
oimgr.grid(row = 2, column = 0)

# Fifth Column
fifthFrame = tk.Frame(master, padx=5, pady=5)
fifthFrame.grid(row = 0, column = 4)
labelInput = tk.Label(fifthFrame, text = 'Error')
labelInput.grid(row = 0, column = 0)
oimg_label4 = tk.Label(fifthFrame, height=20, width=20)
oimg_label4.grid(row = 1, column = 0)
Qapp = tk.Button(fifthFrame, text='Quit', command=master.destroy)
Qapp.grid(row = 2, column = 0)

master.mainloop()


# In[ ]:




