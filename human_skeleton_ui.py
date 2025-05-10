import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import os
from datetime import datetime
from human_skeleton_processor import HumanSkeletonProcessor

class HumanSkeletonUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Skeleton Estimation")
        self.root.geometry("1200x800")
        
        # Initialize the processor
        self.processor = HumanSkeletonProcessor()
        
        # Variables
        self.cap = cv2.VideoCapture(0)
        self.is_using_camera = True
        self.selected_image = None
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for camera and buttons
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Camera frame
        self.camera_frame = ttk.LabelFrame(left_panel, text="Camera Feed")
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Buttons frame
        buttons_frame = ttk.Frame(left_panel)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.select_bg_btn = ttk.Button(buttons_frame, text="Select Background from Device", command=self.select_background)
        self.select_bg_btn.pack(side=tk.LEFT, padx=5)
        
        self.select_person_btn = ttk.Button(buttons_frame, text="Select Person from Device", command=self.select_person)
        self.select_person_btn.pack(side=tk.LEFT, padx=5)
        self.select_person_btn.config(state=tk.DISABLED)
        
        self.capture_bg_btn = ttk.Button(buttons_frame, text="Capture Background", command=self.capture_background)
        self.capture_bg_btn.pack(side=tk.LEFT, padx=5)
        
        self.capture_fg_btn = ttk.Button(buttons_frame, text="Capture Person", command=self.capture_foreground)
        self.capture_fg_btn.pack(side=tk.LEFT, padx=5)
        self.capture_fg_btn.config(state=tk.DISABLED)
        
        self.process_btn = ttk.Button(buttons_frame, text="Process & Estimate Skeleton", command=self.process_images)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        self.process_btn.config(state=tk.DISABLED)
        
        # Right panel for processed images
        self.right_panel = ttk.Frame(main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Start camera
        self.update_camera()
        
    def update_camera(self):
        if not self.is_using_camera:
            return
            
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.camera_label.config(image=img)
            self.camera_label.image = img
        self.root.after(10, self.update_camera)
        
    def select_background(self):
        file_path = filedialog.askopenfilename(
            title="Select Background Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            # Stop camera feed if this is the first selection
            if self.is_using_camera:
                self.is_using_camera = False
                
            # Load the selected image
            selected_image = cv2.imread(file_path)
            if selected_image is not None:
                # Set as background image in processor
                self.processor.set_background(selected_image)
                
                # Display the image
                display_img = selected_image.copy()
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                
                # Update the display
                img = Image.fromarray(display_img)
                img = ImageTk.PhotoImage(image=img)
                self.camera_label.config(image=img)
                self.camera_label.image = img
                
                # Update camera frame title
                self.camera_frame.config(text="Selected Background Image")
                
                # Enable the person selection and foreground capture buttons
                self.select_person_btn.config(state=tk.NORMAL)
                self.capture_fg_btn.config(state=tk.NORMAL)
                
                messagebox.showinfo("Success", "Background image loaded successfully!")
            else:
                messagebox.showerror("Error", "Failed to load the background image!")
    
    def select_person(self):
        file_path = filedialog.askopenfilename(
            title="Select Person Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            # Stop camera feed if this is the first selection
            if self.is_using_camera:
                self.is_using_camera = False
                
            # Load the selected image
            selected_image = cv2.imread(file_path)
            if selected_image is not None:
                # Set as foreground image in processor
                self.processor.set_foreground(selected_image)
                
                # Display the image
                display_img = selected_image.copy()
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                
                # Update the display
                img = Image.fromarray(display_img)
                img = ImageTk.PhotoImage(image=img)
                self.camera_label.config(image=img)
                self.camera_label.image = img
                
                # Update camera frame title
                self.camera_frame.config(text="Selected Person Image")
                
                # Enable the process button
                self.process_btn.config(state=tk.NORMAL)
                
                messagebox.showinfo("Success", "Person image loaded successfully!")
            else:
                messagebox.showerror("Error", "Failed to load the person image!")
        
    def capture_background(self):
        ret, frame = self.cap.read()
        if ret:
            self.processor.set_background(frame)
            messagebox.showinfo("Success", "Background captured successfully!")
            self.capture_fg_btn.config(state=tk.NORMAL)
            self.select_person_btn.config(state=tk.NORMAL)
            
    def capture_foreground(self):
        ret, frame = self.cap.read()
        if ret:
            self.processor.set_foreground(frame)
            messagebox.showinfo("Success", "Person image captured successfully!")
            self.process_btn.config(state=tk.NORMAL)
            
    def process_images(self):
        # Clear previous results
        for widget in self.right_panel.winfo_children():
            widget.destroy()
        
        # Process images using the processor class
        success = self.processor.process_images()
        
        if not success:
            messagebox.showerror("Error", "Please provide both background and person images!")
            return
            
        # Display results
        self.display_results()
        
    def display_results(self):
        try:
            # Get processed images from processor
            processed_images = self.processor.get_processed_images()
            
            # Create a notebook for tabs
            notebook = ttk.Notebook(self.right_panel)
            notebook.pack(fill=tk.BOTH, expand=True)
            
            # Create tabs for each processing step
            for step_name, img in processed_images.items():
                tab = ttk.Frame(notebook)
                notebook.add(tab, text=step_name)
                
                # Convert to RGB if needed
                if len(img.shape) == 2:
                    display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    display_img = img
                    
                # Create matplotlib figure
                fig = plt.Figure(figsize=(6, 4), dpi=100)
                ax = fig.add_subplot(111)
                ax.imshow(display_img)
                ax.set_title(step_name)
                ax.axis('off')
                
                # Add figure to tab
                canvas = FigureCanvasTkAgg(fig, master=tab)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            print(f"Error displaying results: {e}")
            # Force update the main window
            self.root.update()
        
    def on_closing(self):
        if self.is_using_camera and self.cap.isOpened():
            self.cap.release()
        self.root.destroy() 