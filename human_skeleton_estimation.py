import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime
from skimage.morphology import skeletonize, thin
from skimage.util import invert
from skimage.filters import gaussian
from sklearn.cluster import KMeans

def preprocess_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed

class HumanSkeletonEstimator:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Skeleton Estimation")
        self.root.geometry("1200x800")
        
        # Variables
        self.background_image = None
        self.foreground_image = None
        self.processed_images = {}
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
                # Set as background image
                self.background_image = selected_image
                
                # # Display the image
                # display_img = selected_image.copy()
                # display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                
                # # Update the display
                # img = Image.fromarray(display_img)
                # img = ImageTk.PhotoImage(image=img)
                # self.camera_label.config(image=img)
                # self.camera_label.image = img
                
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
                # Set as foreground image
                self.foreground_image = selected_image
                
                # # Display the image
                # display_img = selected_image.copy()
                # display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                
                # # Update the display
                # img = Image.fromarray(display_img)
                # img = ImageTk.PhotoImage(image=img)
                # self.camera_label.config(image=img)
                # self.camera_label.image = img
                
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
            self.background_image = frame
            messagebox.showinfo("Success", "Background captured successfully!")
            self.capture_fg_btn.config(state=tk.NORMAL)
            self.select_person_btn.config(state=tk.NORMAL)
            
    def capture_foreground(self):
        ret, frame = self.cap.read()
        if ret:
            self.foreground_image = frame
            messagebox.showinfo("Success", "Person image captured successfully!")
            self.process_btn.config(state=tk.NORMAL)
            
    def process_images(self):
        if self.background_image is None or self.foreground_image is None:
            messagebox.showerror("Error", "Please provide both background and person images!")
            return
            
        # Clear previous results
        for widget in self.right_panel.winfo_children():
            widget.destroy()
        
        self.processed_images = {}
        
        # Extract silhouette by background subtraction
        diff = cv2.absdiff(self.background_image, self.foreground_image)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Negation
        negated = cv2.bitwise_not(gray_diff)
        self.processed_images["1. Negation"] = negated
        
        # Step 2: Threshold (2*mean)
        mean_val = np.mean(gray_diff)
        _, thresh = cv2.threshold(gray_diff, 2*mean_val, 255, cv2.THRESH_BINARY)
        self.processed_images["2. Threshold (2*mean)"] = thresh
        
        # Step 3: Silhouette Extraction
        kernel = np.ones((5, 5), np.uint8)
        silhouette = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_OPEN, kernel)
        self.processed_images["3. Silhouette Extraction"] = silhouette
        
        # Step 4: Silhouette Negation
        silhouette_neg = cv2.bitwise_not(silhouette)
        self.processed_images["4. Silhouette Negation"] = silhouette_neg
        
        # Step 5: Silhouette Smoothing with Gaussian Filter
        silhouette_smooth = gaussian(silhouette/255.0, sigma=2)
        silhouette_smooth = (silhouette_smooth * 255).astype(np.uint8)
        self.processed_images["5. Gaussian Smoothing"] = silhouette_smooth
        
        # Step 6: Threshold (based on histogram)
        hist = cv2.calcHist([silhouette_smooth], [0], None, [256], [0, 256])
        hist_peak = np.argmax(hist)
        _, thresh_hist = cv2.threshold(silhouette_smooth, hist_peak, 255, cv2.THRESH_BINARY)
        self.processed_images["6. Histogram Threshold"] = thresh_hist

        # Bước 2: Làm đầy ảnh bị thủng
        thresh_filled = preprocess_mask(thresh_hist)

        
        # Step 7: Thinning
        # Convert to binary (0 and 1)
        binary = thresh_filled > 0
        thinned = thin(binary)
        thinned_img = (thinned * 255).astype(np.uint8)
        self.processed_images["7. Thinning"] = thinned_img
        
        # Step 8: Skeletonization
        skeleton = skeletonize(binary)
        skeleton_img = (skeleton * 255).astype(np.uint8)
        self.processed_images["8. Skeletonization"] = skeleton_img
        
        # Step 9: Interest Points Detection
        interest_points_img = self.foreground_image.copy()
        joints = self.find_interest_points(thinned)
        
        # Draw joints on original image
        for point in joints:
            cv2.circle(interest_points_img, (point[1], point[0]), 5, (0, 255, 0), -1)
            
        # Connect joints to form skeleton
        skeleton_img_color = cv2.cvtColor(self.foreground_image, cv2.COLOR_BGR2RGB)
        # self.draw_skeleton(skeleton_img_color, joints)

        skeleton_img_color = self.draw_skeleton2(thinned)
        
        self.processed_images["9. Interest Points"] = cv2.cvtColor(interest_points_img, cv2.COLOR_BGR2RGB)
        self.processed_images["10. Final Skeleton"] = skeleton_img_color
        
        # Display results
        self.display_results()
        
    def find_interest_points(self, skeleton):
        # Find endpoints and junction points
        points = []
        
        # Create a kernel for convolution to detect interest points
        for i in range(1, skeleton.shape[0]-1):
            for j in range(1, skeleton.shape[1]-1):
                if skeleton[i, j]:
                    # Get 3x3 neighborhood
                    patch = skeleton[i-1:i+2, j-1:j+2].copy()
                    patch[1, 1] = 0  # Remove center
                    neighbors = np.sum(patch)
                    
                    # Endpoints have 1 neighbor, junctions have >2 neighbors
                    if neighbors == 1 or neighbors > 2:
                        points.append((i, j))
                        
        return points
    
    def draw_skeleton2(self, thinned, k=15):
        # 1. Lấy tất cả toạ độ pixel trắng (giá trị 255)
        yx = np.column_stack(np.where(thinned > 0))  # (row, col) format → (y, x)

        if len(yx) < k:
            print("Không đủ điểm để clustering.")
            return thinned

        # 2. Clustering thành 15 cụm
        kmeans = KMeans(n_clusters=k, random_state=0).fit(yx)
        centers = kmeans.cluster_centers_.astype(int)

        # 3. Hiển thị ảnh và vẽ các điểm
        # Convert boolean thinned to uint8 (0-255) before conversion to BGR
        thinned_uint8 = (thinned * 255).astype(np.uint8)
        skeleton_img = cv2.cvtColor(thinned_uint8, cv2.COLOR_GRAY2BGR)

        # 4. Vẽ 15 điểm keypoint
        for (y, x) in centers:
            cv2.circle(skeleton_img, (x, y), 4, (0, 0, 255), -1)

        # 5. Nối các điểm keypoint (dựa trên thứ tự index — cần sắp thủ công)
        # => Bạn có thể điều chỉnh thứ tự theo giải phẫu: head, shoulders, elbows, wrists, torso, hips, knees, ankles
        connections = [
            (0, 1), (1, 2), (2, 3),  # left arm
            (1, 4), (4, 5), (5, 6),  # right arm
            (1, 7), (7, 8), (8, 9),  # torso to left leg
            (7, 10), (10, 11), (11, 12),  # torso to right leg
            (1, 13), (13, 14)  # neck and head
        ]

        for i, j in connections:
            if i < k and j < k:
                pt1 = tuple(centers[i][::-1])  # (x, y)
                pt2 = tuple(centers[j][::-1])
                # cv2.line(skeleton_img, pt1, pt2, (0, 255, 0), 2)

        # Return the image with skeleton drawn rather than showing it directly
        return cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2RGB)
    
    def draw_skeleton(self, image, joints):
        if len(joints) < 5:
            return

        # Sắp xếp theo y (từ trên xuống dưới)
        joints_sorted = sorted(joints, key=lambda p: p[0])
        
        # Lấy điểm cao nhất làm "head"
        head = joints_sorted[0]

        # Giả sử các điểm gần head là vai và cổ
        upper_body = joints_sorted[1:5]
        
        # Giả sử các điểm giữa là hông và gối
        middle_body = joints_sorted[5:9]

        # Giả sử các điểm thấp nhất là bàn chân
        lower_body = joints_sorted[9:]

        # Giả lập mapping (nên cải tiến bằng gán nhãn tốt hơn)
        skeleton_points = {
            "head": head,
            "neck": upper_body[0],
            "left_shoulder": upper_body[1],
            "right_shoulder": upper_body[2],
            "torso": middle_body[0],
            "left_hip": middle_body[1],
            "right_hip": middle_body[2],
            "left_knee": lower_body[0],
            "right_knee": lower_body[1],
            "left_foot": lower_body[2],
            "right_foot": lower_body[3] if len(lower_body) > 3 else lower_body[2],
        }

        connections = [
            ("head", "neck"), ("neck", "left_shoulder"), ("neck", "right_shoulder"),
            ("neck", "torso"), ("torso", "left_hip"), ("torso", "right_hip"),
            ("left_hip", "left_knee"), ("left_knee", "left_foot"),
            ("right_hip", "right_knee"), ("right_knee", "right_foot"),
        ]

        # Vẽ skeleton
        for part1, part2 in connections:
            if part1 in skeleton_points and part2 in skeleton_points:
                pt1 = skeleton_points[part1]
                pt2 = skeleton_points[part2]
                cv2.line(image, (pt1[1], pt1[0]), (pt2[1], pt2[0]), (0, 0, 255), 2)



        # # Simple skeleton drawing - connect nearby joints
        # if len(joints) < 2:
        #     return
            
        # # Sort joints roughly by y-coordinate (top to bottom)
        # joints_sorted = sorted(joints, key=lambda p: p[0])
        
        # # Draw lines between points that are likely to be connected
        # for i in range(len(joints_sorted)-1):
        #     for j in range(i+1, min(i+4, len(joints_sorted))):
        #         # Calculate Euclidean distance
        #         dist = np.sqrt((joints_sorted[i][0]-joints_sorted[j][0])**2 + 
        #                       (joints_sorted[i][1]-joints_sorted[j][1])**2)
                
        #         # Connect points that are reasonably close
        #         if dist < 100:  # Threshold for connection
        #             cv2.line(image, 
        #                     (joints_sorted[i][1], joints_sorted[i][0]), 
        #                     (joints_sorted[j][1], joints_sorted[j][0]), 
        #                     (0, 0, 255), 2)
        
    def display_results(self):
        try:
            # Create a notebook for tabs
            notebook = ttk.Notebook(self.right_panel)
            notebook.pack(fill=tk.BOTH, expand=True)
            
            # Create tabs for each processing step
            for step_name, img in self.processed_images.items():
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


if __name__ == "__main__":
    root = tk.Tk()
    app = HumanSkeletonEstimator(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop() 