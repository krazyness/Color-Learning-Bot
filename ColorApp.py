import numpy as np
from sklearn.linear_model import SGDClassifier
import tkinter as tk
from tkinter import messagebox
import os
import joblib

X = np.array([[255,0,0],[0,255,0],[0,0,255]])
y = np.array([0,1,2])

if os.path.exists("color_model.pkl"):
    clf = joblib.load("color_model.pkl")
else:
    clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)
    clf.partial_fit(X, y, classes=[0,1,2])

color_names = ["red", "green", "blue"]

def valid_rgb(rgb):
    return len(rgb) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in rgb)

class ColorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Color Classifier")
        self.geometry("300x250")

        tk.Label(self, text="Enter RGB values (0-255):").pack()
        self.r_entry = tk.Entry(self, width=5)
        self.g_entry = tk.Entry(self, width=5)
        self.b_entry = tk.Entry(self, width=5)
        self.r_entry.pack(side=tk.LEFT, padx=(30,0))
        self.g_entry.pack(side=tk.LEFT)
        self.b_entry.pack(side=tk.LEFT)

        self.show_btn = tk.Button(self, text="Show Color & Predict", command=self.show_and_predict)
        self.show_btn.pack(pady=10)

        self.canvas = tk.Canvas(self, width=100, height=100, bg='white')
        self.canvas.pack(pady=5)

        self.pred_label = tk.Label(self, text="")
        self.pred_label.pack()

        self.correction_var = tk.StringVar(self)
        self.correction_menu = tk.OptionMenu(self, self.correction_var, *color_names)
        self.correction_menu.pack()
        self.correction_menu.config(state="disabled")

        self.learn_btn = tk.Button(self, text="Correct and Learn", command=self.correct_and_learn, state="disabled")
        self.learn_btn.pack(pady=5)

    def show_and_predict(self):
        try:
            rgb = [int(self.r_entry.get()), int(self.g_entry.get()), int(self.b_entry.get())]
            if not valid_rgb(rgb):
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid Input", "Please enter three numbers between 0 and 255.")
            return

        self.canvas.config(bg='#%02x%02x%02x' % tuple(rgb))

        pred = clf.predict([rgb])[0]
        self.pred_label.config(text=f"I think it's {color_names[pred]}!")
        self.correction_var.set(color_names[pred])
        self.correction_menu.config(state="normal")
        self.learn_btn.config(state="normal")
        self.last_rgb = rgb

    def correct_and_learn(self):
        label = color_names.index(self.correction_var.get())
        for _ in range(10):
            clf.partial_fit([self.last_rgb], [label])
        joblib.dump(clf, "color_model.pkl")
        messagebox.showinfo("Learned", f"Thanks! I've learned that this is {color_names[label]}.")

if __name__ == "__main__":
    app = ColorApp()
    app.mainloop()