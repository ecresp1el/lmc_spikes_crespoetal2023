import os
import tkinter as tk
from tkinter import filedialog

os.environ['TK_SILENCE_DEPRECATION'] = '1'

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

print(file_path)