from tkinter import *
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from matplotlib import pyplot as plt

blood=[False,False,False,False]
q = 1
f = 0
v = 0
p1 = ''
p2 = ''
p3 = ''
p4 = ''


class Login(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    def init_window(self):

        # Code Segment for Icon and title

        self.configure(background='powder blue')
        self.pack(fill=BOTH, expand=1)
        self.master.title("Blood Group Detection System")
        self.master.iconbitmap('D:\Blood-group-Detection-using-python\Blood.ico')

        # Code segment for dropdown menu

        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        stage = Menu(menu)
        file.add_cascade(label="Process", menu=stage)
        file.add_command(label="Restart",command=self.restart)
        file.add_command(label="Exit",command=self.quit)
        menu.add_cascade(label="File", menu=file)

        stage.add_command(label="Process 1: Green Plane Extraction", command=self.gp)
        stage.add_command(label="Process 2: Auto Threshold", command=self.autothresh)
        stage.add_command(label="Process 3: Adaptive Threshold:Ni Black", command=self.Adapthresh)
        stage.add_command(label="Process 4: Morphology: Fill Holes", command=self.Fill_holes)
        stage.add_command(label="Process 5: Advanced Morphology: Remove small objects", command=self.Remove_small_objects)
        stage.add_command(label="Process 6: Histogram", command=self.Histogram)
        stage.add_command(label="Process 7: Quantification", command=self.HSV_Luminance)

        # Code segment for labels
        l1 = Label(self, text="Reagent Anti-A", font=("Helvetica", 12))
        l2 = Label(self, text="Reagent Anti-B", font=("Helvetica", 12))
        l3 = Label(self, text="Reagent Anti-D", font=("Helvetica", 12))
        l4 = Label(self, text="Control Reagent", font=("Helvetica", 12))
        l1.place(x=160, y=475)
        l2.place(x=480, y=475)
        l3.place(x=780, y=475)
        l4.place(x=1070, y=475)

        # Code segment for buttons
        e1 = Button(self, text="Choose Image", command=self.imagesel1)
        e2 = Button(self, text="Choose Image", command=self.imagesel2)
        e3 = Button(self, text="Choose Image", command=self.imagesel3)
        e4 = Button(self, text="Choose Image", command=self.imagesel4)
        self.ep = Button(self, text="Process", font=("Helvetica", 12), fg='red', relief=SUNKEN)
        self.ep.place(x=650, y=575)
        e1.place(x=170, y=500)
        e2.place(x=490, y=500)
        e3.place(x=790, y=500)
        e4.place(x=1080, y=500)

    def quit(self):
        global q
        q = 0
        root.destroy()


    def restart(self):
        global q
        q = 1
        root.destroy()

    def  message(self,q):
        messagebox.showinfo("Result",q+"Confirmed")

    def start1(self):
        self.start(p1,"Anti A")
        self.start2()


    def imagesel1(self):
        global v
        v += 1
        s = filedialog.askopenfilename()
        x = ""
        i = len(s)-1
        while s[i] != '/':
            x += s[i]
            i -= 1
        global p1
        p1 = x[::-1]
        self.p = Image.open(x[::-1])
        r = self.p.resize((300,425),Image.ANTIALIAS)
        i = ImageTk.PhotoImage(r)
        l = Label(self, image=i)
        l.Image = i
        l.place(x=75, y=50)
        if v == 4:
            self.ep.configure(relief=RAISED, fg='green', command=self.start1)

    def start2(self):
        self.start(p2, "Anti B")
        self.start3()



    def imagesel2(self):
        global v, p2
        v += 1
        s = filedialog.askopenfilename()
        x = ""
        i = len(s)-1
        while s[i] != '/':
            x += s[i]
            i -= 1
        p2 = x[::-1]
        self.p = Image.open(x[::-1])
        r = self.p.resize((300, 425), Image.ANTIALIAS)
        i = ImageTk.PhotoImage(r)
        l = Label(self, image=i)
        l.Image = i
        l.place(x=375, y=50)
        if v == 4:
            self.ep.configure(relief=RAISED, fg='green', command=self.start1)

    def start3(self):
        self.start(p3, "Anti D")
        self.start4()


    def imagesel3(self):
        global v, p3
        v += 1
        s = filedialog.askopenfilename()
        x = ""
        i = len(s)-1
        while s[i] != '/':
            x += s[i]
            i -= 1
        p3 = x[::-1]
        self.p = Image.open(x[::-1])
        r = self.p.resize((300,425),Image.ANTIALIAS)
        i = ImageTk.PhotoImage(r)
        l = Label(self, image=i)
        l.Image = i
        l.place(x=675, y=50)
        if v == 4:
            self.ep.configure(relief=RAISED, fg='green', command=self.start1)

    def start4(self):
        self.start(p4, "Control")
        self.check()

    def imagesel4(self):
        global v, p4
        v += 1
        s = filedialog.askopenfilename()
        x = ""
        i = len(s)-1
        while s[i] != '/':
            x += s[i]
            i -= 1
        p4 = x[::-1]
        self.p = Image.open(x[::-1])
        r = self.p.resize((300, 425), Image.ANTIALIAS)
        i = ImageTk.PhotoImage(r)
        l = Label(self, image=i)
        l.Image = i
        l.place(x=975, y=50)
        if v == 4:
            self.ep.configure(relief=RAISED, fg='green', command=self.start1)

    def process1(self, p,r):  # Extracting the Green plane
        img = cv2.imread(p)
        gi = img[:, :, 1]
        cv2.imwrite("p1"+r+".png", gi)
        return gi

    def process2(self, p,r):  # Obtaining the threshold
        gi = self.process1(p,r)
        _, th = cv2.threshold(gi, 0, 255, cv2.THRESH_OTSU)
        cv2.imwrite("p2"+r+".png", th)

    def process3(self, p,r):  # Obtaining Ni black image
        img = cv2.imread('p2'+r+'.png', 0)
        th4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 14)
        cv2.imwrite("p3"+r+".png", th4)

    def process4(self,r):  # Morphology: fill holes
        gi = cv2.imread('p3'+r+'.png', cv2.IMREAD_GRAYSCALE)
        th, gi_th = cv2.threshold(gi, 220, 255, cv2.THRESH_BINARY_INV)
        gi_floodFill=gi_th.copy()
        h, w = gi_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(gi_floodFill, mask, (0, 0), 255)
        gi_floodFill_inv = cv2.bitwise_not(gi_floodFill)
        gi_out = gi_th | gi_floodFill_inv
        cv2.imwrite('p4'+r+'.png', gi_out)

    def process5(self,r):  # Morphing To eliminate small objects
        img = cv2.imread('p4'+r+'.png')
        kernel = np.ones((5, 5), np.uint8)
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite('p5'+r+'.png', close)

    def process7(self,r):  #Histogram
        img = cv2.imread('p5'+r+'.png', 0)
        img2 = cv2.imread('p1'+r+'.png', 0)
        mask = np.ones(img.shape[:2], np.uint8)
        hist = cv2.calcHist([img2], [0], mask, [256], [0, 256])
        min = 1000
        max = 0
        n = 0
        s = 0
        ss = 0
        for x, y in enumerate(hist):
            if y > max:
                max = y
            if y < min:
                min = y
            s += y
            n += 1

        mean = s/n
        for x, y in enumerate(hist):
            ss += (y-mean)**2
        ss /= n
        sd = abs(ss)**0.5
        print(r,"-",sd,"\n")
        if sd < 580:
            return 1
        else:
            return 0


    def start(self, p,r):
        global blood
        self.process1(p,r)
        self.process2(p,r)
        self.process3(p,r)
        self.process4(r)
        self.process5(r)
        a = self.process7(r)
        print(a," - ",r)
        if a == 1:
            if r == "Anti A":
                blood[0]=True
            elif r == "Anti B":
                blood[1]=True
            elif r == "Anti D":
                blood[2]=True
            elif r == "Control":
                blood[3]=True

    def check(self):
        if blood[3]==True:
            self.message("Invalid")
        elif blood[0] is False and blood[1] is False and blood[2] is True and blood[3] is False:
            self.message("O+")
        elif blood[0] is False and blood[1] is False and blood[2] is False and blood[3] is False:
            self.message("O-")
        elif blood[0] is True and blood[1] is False and blood[2] is True and blood[3] is False:
            self.message("A+")
        elif blood[0] is True and blood[1] is False and blood[2] is False and blood[3] is False:
            self.message("A-")
        elif blood[0] is False and blood[1] is True and blood[2] is True and blood[3] is False:
            self.message("B+")
        elif blood[0] is False and blood[1] is True and blood[2] is False and blood[3] is False:
            self.message("B-")
        elif blood[0] is True and blood[1] is True and blood[2] is True and blood[3] is False:
            self.message("AB+")
        elif blood[0] is True and blood[1] is True and blood[2] is False and blood[3] is False:
            self.message("AB-")


    def gp(self):
        im1 = cv2.imread('p1Anti A.png')
        cv2.imshow('Anti-A',im1)
        im2 = cv2.imread('p1Anti B.png')
        cv2.imshow('Anti-B', im2)
        im3 = cv2.imread('p1Anti D.png')
        cv2.imshow('Anti-D', im3)
        im4 = cv2.imread('p1Control.png')
        cv2.imshow('Control', im4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def autothresh(self):
        im1 = cv2.imread('p2Anti A.png')
        cv2.imshow('Anti-A', im1)
        im2 = cv2.imread('p2Anti B.png')
        cv2.imshow('Anti-B', im2)
        im3 = cv2.imread('p2Anti D.png')
        cv2.imshow('Anti-D', im3)
        im4 = cv2.imread('p2Control.png')
        cv2.imshow('Control', im4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Adapthresh(self):
        im1 = cv2.imread('p3Anti A.png')
        cv2.imshow('Anti-A', im1)
        im2 = cv2.imread('p3Anti B.png')
        cv2.imshow('Anti-B', im2)
        im3 = cv2.imread('p3Anti D.png')
        cv2.imshow('Anti-D', im3)
        im4 = cv2.imread('p3Control.png')
        cv2.imshow('Control', im4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Fill_holes(self):
        im1 = cv2.imread('p4Anti A.png')
        cv2.imshow('Anti-A', im1)
        im2 = cv2.imread('p4Anti B.png')
        cv2.imshow('Anti-B', im2)
        im3 = cv2.imread('p4Anti D.png')
        cv2.imshow('Anti-D', im3)
        im4 = cv2.imread('p4Control.png')
        cv2.imshow('Control', im4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Remove_small_objects(self):
        im1 = cv2.imread('p5Anti A.png')
        cv2.imshow('Anti-A', im1)
        im2 = cv2.imread('p5Anti B.png')
        cv2.imshow('Anti-B', im2)
        im3 = cv2.imread('p5Anti D.png')
        cv2.imshow('Anti-D', im3)
        im4 = cv2.imread('p5Control.png')
        cv2.imshow('Control', im4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Histogram(self):
        img1 = cv2.imread('p5Anti A.png', 0)
        img2 = cv2.imread('p5Anti B.png', 0)
        img3 = cv2.imread('p5Anti D.png', 0)
        img4 = cv2.imread('p5Control.png', 0)
        plt.hist(img1.ravel(), 256, [0, 256])
        plt.show()
        plt.hist(img2.ravel(), 256, [0, 256])
        plt.show()
        plt.hist(img3.ravel(), 256, [0, 256])
        plt.show()
        plt.hist(img4.ravel(), 256, [0, 256])
        plt.show()

    def HSV_Luminance(self):
        img1 = cv2.imread(p1)
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv1, 0)

        img2 = cv2.imread(p2)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv2, 0)

        img3 = cv2.imread(p3)
        hsv3 = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv3, 0)

        img4 = cv2.imread(p4)
        hsv4 = cv2.cvtColor(img4, cv2.COLOR_BGR2HSV)
        cv2.imshow(hsv4, 0)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def stp_full(self,event=None):
        root.attributes("-fullscreen", False)
        root.geometry("1020x720")




while(1):
    if q == 0:
        break
    else:
        root = Tk()
        root.attributes("-fullscreen",True)
        app = Login(root)
        root.bind("<Escape>", app.stp_full)
        root.mainloop()
