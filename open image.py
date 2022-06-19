from cProfile import label
from pickle import FROZENSET
from tkinter import *
from tkinter import filedialog
import os
import tkinter as tk
from PIL import  Image,ImageTk
from matplotlib.image import thumbnail
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

model = load_model('C:/Users/DELL/Documents/People_CNN/Model123.h5')

classes = ['Ear', 'Elbow', 'Eye', 'Foot', 'Hand', 'Knee', 'Nose', 'Shoulders']

def showimage():
    global fln
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image File", filetypes=(("JPG File","*.jpg"),("PNG File","*.png"), ("ALL Files","*.*")))
    img = Image.open(fln)
    img.thumbnail((200,200))
    img = ImageTk.PhotoImage(img)
    lbl3.configure(image= img)
    lbl3.image = img

def recognize():
    global lbl1
    global lbl2
    img_path= fln
    img=plt.imread(img_path)
    print ('Input image shape is ', img.shape)
    img=cv2.resize(img, (150,150)) 
    print ('the resized image has shape ', img.shape)
    plt.axis('off')
    plt.imshow(img)
   
    img=np.expand_dims(img, axis=0)
    print ('image shape after expanding dimensions is ',img.shape)
    pred=model.predict(img)
    print ('the shape of prediction is ', pred.shape)
    index=np.argmax(pred[0])
    klass=classes[index]
    probability=pred[0][index]*100
    print(f'the image is predicted as being {klass} with a probability of {probability:6.2f} %')
    lbl1 = Label(root,text = f"Nhận Diện: {klass}" , fg= "green", font=("Arial", 20))
    lbl1.pack(pady= 20)
    lbl2 = Label(root,text = f"Độ chính xác: {probability:6.2f} %" , fg= "blue", font=("Arial", 20))
    lbl2.pack(pady= 20)
    return

def clear():
    lbl1.after(500, lbl1.destroy())
    lbl2.after(500, lbl2.destroy())
    return

root = Tk()
root.title("Recognize Body Parts")
root.geometry("450x520")
# lbl1 = Label(root)
# lbl2 = Label(root)

frm = Frame(root)
frm.pack(side=BOTTOM, padx=15, pady=15)

lbl = Label(root,
            text = "Recognize Body Parts", 
            fg= "red", 
            font=("Arial", 30), 
            background="black"
            )
lbl.pack(padx=10, pady= 10)

lbl3 = Label(root)
lbl3.pack()

btn = Button(frm,text = "Browser Image", command= showimage)
btn.pack(side=tk.LEFT,padx= 15)

btn1 = Button(frm,text = "Recognize", command= recognize)
btn1.pack(side=tk.LEFT,padx= 15, pady = 10)

btn3 = Button(frm,text = "Clear", command= clear )
btn3.pack(side=tk.LEFT,padx=20)

btn2 = Button(frm,text = "EXIT", command=lambda: exit())
btn2.pack(side=tk.LEFT,padx=20)


root.mainloop()