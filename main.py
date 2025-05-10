import tkinter as tk
from human_skeleton_ui import HumanSkeletonUI

if __name__ == "__main__":
    root = tk.Tk()
    app = HumanSkeletonUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop() 