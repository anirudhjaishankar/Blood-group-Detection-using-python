from tkinter import *
from tkinter import filedialog,messagebox
from PIL import Image,ImageTk


class login(Frame):
    def __init__(self,master=None):
        Frame.__init__(self,master)
        self.master=master
        self.init_window()

    def init_window(self):

        self.pack(fill=BOTH,expand=1)
        self.master.title("Blood Group Detection System")
        self.master.iconbitmap('F:\Projects\Blood group detection using python\Blood.ico')

        e1=Button(root,text="Choose Image",command=self.imagesel)
        e2=Button(root,text="Process")
        e2.pack(side=BOTTOM)
        e1.pack(side=BOTTOM)

    def message(self):
        messagebox.showinfo("Result", "Stupid Face")

    def imagesel(self):
        s=filedialog.askopenfilename()
        x=""
        i=len(s)-1
        while s[i] != '/':
            x+=s[i]
            i-=1

        self.p= Image.open(x[::-1])
        r=self.p.resize((250,300),Image.ANTIALIAS)
        for j in range(0,5):
            i=ImageTk.PhotoImage(r)
            l=Label(self,image=i)
            l.Image=i
            l.place(x=50+(j*250),y=50)

        for j in range(0, 5):
            i = ImageTk.PhotoImage(r)
            l = Label(self, image=i)
            l.Image = i
            l.place(x=50 + (j * 250), y=350)
        self.message()








root=Tk()
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
app=login(root)
root.mainloop()