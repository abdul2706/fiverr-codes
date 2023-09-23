import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

WIDTH, HEIGHT = 1400, 600
HIST_PLOT_LABELS = {1: 'Gray', 3: ['Red', 'Green', 'Blue']}

def show_histogram(matrices):
    channels = len(matrices)

    # Toplevel object which will be treated as a new window for showing histogram
    hist_root = Toplevel(root)
    hist_root.title('Image Histogram')
    hist_root.geometry(f'{500 * channels}x500')

    # Add canevas to bottom
    # figure for histogram
    fig_hist = Figure(figsize=(8 * channels, 8))
    canvas_hist = FigureCanvasTkAgg(fig_hist, master=hist_root)
    canvas_hist.get_tk_widget().pack()
    
    for i, matrix in enumerate(matrices):
        ax_hist = fig_hist.add_subplot(1, channels, i + 1)
        ax_hist.hist(matrix.ravel(), bins=256)
        ax_hist.set_xticks(np.arange(0, 257, 32))
        ax_hist.set_xlabel(f'{HIST_PLOT_LABELS[channels][i]} Channel')
        ax_hist.set_ylabel('Frequency')

def show_transformation_function():
    global r1, s1, r2, s2

    # Toplevel object which will be treated as a new window for showing histogram
    function_root = Toplevel(root)
    function_root.title('Transformation Function Plot')
    function_root.geometry(f'500x500')

    # figure for ploting transformation function
    fig_function = Figure(figsize=(8, 8))
    canvas_function = FigureCanvasTkAgg(fig_function, master=function_root)
    canvas_function.get_tk_widget().pack()

    # add canvas to showing transformation function
    ax_piecewise = fig_function.add_subplot(111)
    ax_piecewise.plot([0, r1, r2, 256], [0, s1, s2, 256])
    ax_piecewise.plot([r1, r2], [s1, s2], 'or')
    ax_piecewise.set_xlabel('Input gray level, r')
    ax_piecewise.set_ylabel('Output gray level, s')
    ax_piecewise.set_xticks(np.arange(0, 257, 32))
    ax_piecewise.set_yticks(np.arange(0, 257, 32))
    
def read_image():
    global image, matrices, is_color

    filepath = askopenfilename(filetypes=[("Image Files", "*.*"), ("All Files", "*.*")])
    if not filepath:
        return
        
    image = Image.open(filepath)
    image = np.asarray(image)
    is_color = len(image.shape) == 3
    matrices = cv2.split(image)
    if is_color:
        ax1.imshow(image)
        ax1.set_title('Original')
    else:
        ax1.imshow(image, cmap='gray')
        ax1.set_title('Original')
    canvas.draw()

    show_histogram(matrices)

def process_image(r1, s1, r2, s2):
    global image, matrices, is_color, output_image, outputs

    # r1, s1, r2, s2 = 100, 50, 150, 220
    outputs = []
    for matrix in matrices:
        output = np.copy(matrix)
        for r in range(0, 256):
            if 0 <= r <= r1:
                x1, y1, x2, y2 = 0, 0, r1, s1
            elif r1 < r <= r2:
                x1, y1, x2, y2 = r1, s1, r2, s2
            else:
                x1, y1, x2, y2 = r2, s2, 255, 255

            m = (y2 - y1) / (x2 - x1)
            c = -m * x1 + y1
            s = round(m * r + c)
            output[matrix == r] = s
        outputs.append(output)

    output_image = cv2.merge(outputs) if is_color else outputs[0]
    # print(image.shape, output_image.shape)

    if is_color:
        ax2.imshow(output_image)
        ax2.set_title('Transformed')
    else:
        ax2.imshow(output_image, cmap='gray')
        ax2.set_title('Transformed')
    canvas.draw()

    show_histogram(outputs)

def handle_apply_transformation():
    global input_r1, input_s1, input_r2, input_s2, r1, s1, r2, s2
    r1 = int(input_r1.get())
    s1 = int(input_s1.get())
    r2 = int(input_r2.get())
    s2 = int(input_s2.get())
    # print(f'r1 = {r1}, s1 = {s1}, r2 = {r2}, s2 = {s2}')
    process_image(r1, s1, r2, s2)
    show_transformation_function()

# Create the main frame
root = Tk()
root.geometry(f'{WIDTH}x{HEIGHT}')
frame = Frame(root)
frame.pack()

# Create top frame
topframe = Frame(frame)
topframe.pack(side=TOP)

# Create bottom frame
bottomframe = Frame(frame)
bottomframe.pack(side=BOTTOM)

# Create left frame
leftframe = Frame(bottomframe)
leftframe.pack(side=LEFT)

# Create right frame
rightframe = Frame(bottomframe)
rightframe.pack(side=RIGHT)

# Add a label to top frame
topframe.option_add('*Font', 'Times 18')
mainLabel = Label(topframe, text='Image Processing') #, height=1, width=50)
mainLabel.pack(pady=20)

# Fill leftframe items
leftframe.option_add('*Font', 'Times 18')
button1 = Button(leftframe, text='Load Image', command=read_image)
button1.pack(padx=5, pady=5)
button2 = Button(leftframe, text='Apply Transformation', command=handle_apply_transformation)
button2.pack(padx=5, pady=5)

# add inputs for r1, s1, r2, s2
label_r1 = Label(topframe, textvariable=StringVar(value='r1'), height=1)
label_r1.pack(side='left', pady=10)
input_r1 = Entry(topframe, textvariable=StringVar(None), width=20)
input_r1.pack(side='left', pady=10)

label_s1 = Label(topframe, textvariable=StringVar(value='s1'), height=1)
label_s1.pack(side='left', pady=10)
input_s1 = Entry(topframe, textvariable=StringVar(None), width=20)
input_s1.pack(side='left', pady=10)

label_r2 = Label(topframe, textvariable=StringVar(value='r2'), height=1)
label_r2.pack(side='left', pady=10)
input_r2 = Entry(topframe, textvariable=StringVar(None), width=20)
input_r2.pack(side='left', pady=10)

label_s2 = Label(topframe, textvariable=StringVar(value='s2'), height=1)
label_s2.pack(side='left', pady=10)
input_s2 = Entry(topframe, textvariable=StringVar(None), width=20)
input_s2.pack(side='left', pady=10)

# add canvas to showing image and transformed image
fig = Figure(figsize=(16, 4))
ax1 = fig.add_subplot(121)
ax1.set_xticks([])
ax1.set_yticks([])
ax2 = fig.add_subplot(122)
ax2.set_xticks([])
ax2.set_yticks([])
canvas = FigureCanvasTkAgg(fig, master=rightframe)
canvas.get_tk_widget().pack()

root.title("Image Processing")
root.mainloop()
