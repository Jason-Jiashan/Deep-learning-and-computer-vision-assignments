# main.py
# Main entry point for the project. Provides a simple UI to select modes and parameters.

import tkinter as tk
from tkinter import ttk, filedialog
import threading
from train import train_dqn
from inference import run_inference
from train_detection import train_yolo


def start_task():
    """Callback to start the selected task in a separate thread (to avoid GUI freeze)."""
    mode = mode_var.get()
    if mode == "Train YOLOv5 Detector":
        epochs = int(param1_var.get()) if param1_var.get() else 50
        batch = int(param2_var.get()) if param2_var.get() else 16
        # Run YOLO training (this may take a long time)
        imgsz  = int(param3_var.get()) if param3_var.get() else 640
        threading.Thread(target=train_yolo,args=(epochs, batch, imgsz, 'gui_run')).start()
    elif mode == "Train RL Agent":
        episodes = int(param1_var.get()) if param1_var.get() else 100
        steps = int(param2_var.get()) if param2_var.get() else 100
        threading.Thread(target=train_dqn, args=(episodes, steps)).start()
    elif mode == "Run Simulation (RL Control)":
        steps = int(param1_var.get()) if param1_var.get() else 100
        threading.Thread(target=run_inference, args=(True, None, steps)).start()
    elif mode == "Run Simulation (Fixed Control)":
        steps = int(param1_var.get()) if param1_var.get() else 100
        threading.Thread(target=run_inference, args=(False, None, steps)).start()
    elif mode == "Video Analysis (Detection+Tracking)":
        video_path = param1_var.get()
        use_rl_flag = rl_check_var.get()
        threading.Thread(target=run_inference, args=(use_rl_flag, video_path, 0)).start()

# Set up Tkinter UI
root = tk.Tk()
root.title("TrafficRLProject Interface")
root.geometry("500x300")

# Mode selection dropdown
mode_var = tk.StringVar(value="Train YOLOv5 Detector")
modes = ["Train YOLOv5 Detector", "Train RL Agent", "Run Simulation (RL Control)",
         "Run Simulation (Fixed Control)", "Video Analysis (Detection+Tracking)"]
mode_label = tk.Label(root, text="Select Mode:")
mode_label.pack(pady=5)
mode_dropdown = ttk.Combobox(root, textvariable=mode_var, values=modes, state="readonly")
mode_dropdown.pack(pady=5)

# Parameter inputs
param1_var = tk.StringVar()
param2_var = tk.StringVar()
param3_var = tk.StringVar()
param1_label = tk.Label(root, text="Param1:")
param2_label = tk.Label(root, text="Param2:")
param3_label = tk.Label(root, text="Img Size:")
param1_entry = tk.Entry(root, textvariable=param1_var)
param2_entry = tk.Entry(root, textvariable=param2_var)
param3_entry = tk.Entry(root, textvariable=param3_var)
param1_label.pack()
param1_entry.pack()
param2_label.pack()
param2_entry.pack()
param3_label.pack()
param3_entry.pack()

# Additional checkbox for RL overlay in video mode
rl_check_var = tk.BooleanVar(value=True)
rl_checkbox = tk.Checkbutton(root, text="Use RL Control (for video mode)", variable=rl_check_var)
rl_checkbox.pack(pady=5)

# File selection for video analysis
def choose_file():
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if file_path:
        param1_var.set(file_path)

file_button = tk.Button(root, text="Choose Video File", command=choose_file)
file_button.pack(pady=5)

# Start button
start_button = tk.Button(root, text="Run", command=start_task, bg="green", fg="white")
start_button.pack(pady=10)

# Adjust UI elements based on mode selection
def on_mode_change(event=None):
    mode = mode_var.get()
    for w in (param3_label, param3_entry, rl_checkbox, file_button):
        w.pack_forget()
    if mode == "Train YOLOv5 Detector":
        param1_label.config(text="Epochs:")
        param2_label.config(text="Batch Size:")
        param3_label.config(text="Img Size:")
        param3_label.pack()
        param3_entry.pack()
        rl_checkbox.pack_forget()
        file_button.pack_forget()
    elif mode == "Train RL Agent":
        param1_label.config(text="Episodes:")
        param2_label.config(text="Steps per Ep:")
        rl_checkbox.pack_forget()
        file_button.pack_forget()
    elif mode in ["Run Simulation (RL Control)", "Run Simulation (Fixed Control)"]:
        param1_label.config(text="Simulation Steps:")
        param2_label.config(text="(Unused)")
        param2_var.set("")  # clear second param
        rl_checkbox.pack_forget()
        file_button.pack_forget()
    elif mode == "Video Analysis (Detection+Tracking)":
        param1_label.config(text="Video Path:")
        param2_label.config(text="(Unused)")
        param2_var.set("")
        rl_checkbox.pack(pady=5)
        file_button.pack(pady=5)

mode_dropdown.bind("<<ComboboxSelected>>", on_mode_change)

# Run the GUI loop
on_mode_change()  # set initial form
root.mainloop()
