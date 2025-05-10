import cv2
import numpy as np
from skimage.morphology import skeletonize, thin
from skimage.util import invert
from skimage.filters import gaussian
from sklearn.cluster import KMeans

def preprocess_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed

class HumanSkeletonProcessor:
    def __init__(self):
        self.background_image = None
        self.foreground_image = None
        self.processed_images = {}
        
    def set_background(self, image):
        self.background_image = image
        
    def set_foreground(self, image):
        self.foreground_image = image
    
    def get_processed_images(self):
        return self.processed_images
    
    def process_images(self):
        if self.background_image is None or self.foreground_image is None:
            return False
            
        self.processed_images = {}
        
        # Extract silhouette by background subtraction
        diff = cv2.absdiff(self.background_image, self.foreground_image)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Negation
        negated = cv2.bitwise_not(gray_diff)
        self.processed_images["1. Negation"] = negated
        
        # IMPROVED MASK GENERATION
        # Step 2: Enhanced Thresholding
        # First apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_diff, (5, 5), 0)
        self.processed_images["2.1 Blurred Difference"] = blurred
        
        # Apply Otsu's thresholding instead of fixed threshold
        _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.processed_images["2.2 Otsu Threshold"] = thresh_otsu
        
        # Also apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2)
        
        # Combine both thresholds (take the union)
        combined_thresh = cv2.bitwise_or(thresh_otsu, adaptive_thresh)
        self.processed_images["2.3 Combined Threshold"] = combined_thresh
        
        # Step 3: Advanced Silhouette Extraction
        # Fill holes using morphological closing with a larger kernel
        kernel_close = np.ones((9, 9), np.uint8)
        closed = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel_close)
        
        # Apply morphological opening to remove small noise
        kernel_open = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        # Find contours and keep only the largest (the person)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask with all contours
        silhouette = np.zeros_like(opened)
        
        if contours:
            # Find the largest contour (assumed to be the person)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Fill the largest contour
            cv2.drawContours(silhouette, [largest_contour], 0, 255, -1)
            
            # Additional closing to ensure a solid mask
            silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, kernel_close)
        
        self.processed_images["3. Silhouette Extraction"] = silhouette
        
        # Step 4: Silhouette Negation
        silhouette_neg = cv2.bitwise_not(silhouette)
        self.processed_images["4. Silhouette Negation"] = silhouette_neg
        
        # Step 5: Silhouette Smoothing with Gaussian Filter
        silhouette_smooth = gaussian(silhouette/255.0, sigma=2)
        silhouette_smooth = (silhouette_smooth * 255).astype(np.uint8)
        self.processed_images["5. Gaussian Smoothing"] = silhouette_smooth
        
        # Step 6: Final Mask Preparation
        # Ensure binary mask with no holes
        _, final_mask = cv2.threshold(silhouette_smooth, 127, 255, cv2.THRESH_BINARY)
        final_mask = preprocess_mask(final_mask)  # Fill any remaining holes
        self.processed_images["6. Final Mask"] = final_mask

        # Step 7: Thinning
        # Convert to binary (0 and 1)
        binary = final_mask > 0
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
        skeleton_img_color = self.draw_skeleton2(thinned)
        
        self.processed_images["9. Interest Points"] = cv2.cvtColor(interest_points_img, cv2.COLOR_BGR2RGB)
        self.processed_images["10. Final Skeleton"] = skeleton_img_color
        
        return True
    
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
        # Get all white pixel coordinates
        yx = np.column_stack(np.where(thinned > 0))  # (row, col) format → (y, x)

        if len(yx) < k:
            print("Not enough points for clustering.")
            return thinned

        # Convert boolean thinned to uint8 (0-255) before conversion to BGR
        thinned_uint8 = (thinned * 255).astype(np.uint8)
        skeleton_img = cv2.cvtColor(thinned_uint8, cv2.COLOR_GRAY2BGR)
        
        # Get image dimensions for reference
        height, width = thinned.shape
        
        # Divide the image into regions for body parts
        top_region = int(height * 0.25)      # Head/neck region
        middle_region = int(height * 0.55)   # Torso region
        
        # Divide image horizontally
        left_region = int(width * 0.4)
        right_region = int(width * 0.6)
        
        # ANATOMICAL REGIONS GROUPING
        # Group points into different body parts based on their position
        spine_points = []
        left_arm_points = []
        right_arm_points = []
        left_leg_points = []
        right_leg_points = []
        
        # Filter skeleton points into appropriate body parts
        for y, x in yx:
            # Spine points (center vertical)
            if left_region <= x <= right_region:
                # Head/neck
                if y < top_region:
                    spine_points.append((y, x))
                # Middle spine
                elif y < middle_region:
                    spine_points.append((y, x))
                # Lower spine
                else:
                    spine_points.append((y, x))
            # Left side of the body
            elif x < left_region:
                # Left arm
                if y < middle_region:
                    left_arm_points.append((y, x))
                # Left leg
                else:
                    left_leg_points.append((y, x))
            # Right side of the body
            else:
                # Right arm
                if y < middle_region:
                    right_arm_points.append((y, x))
                # Right leg
                else:
                    right_leg_points.append((y, x))
        
        # Function to select n evenly distributed points from a set
        def get_distributed_points(points, n):
            if len(points) == 0:
                return []
            if len(points) <= n:
                return points
            
            # Sort points by y-coordinate (top to bottom)
            points = sorted(points, key=lambda p: p[0])
            
            # Select evenly distributed points
            indices = np.linspace(0, len(points) - 1, n, dtype=int)
            return [points[i] for i in indices]
        
        # Get 3 points for each body part
        spine_keypoints = get_distributed_points(spine_points, 3)
        left_arm_keypoints = get_distributed_points(left_arm_points, 3)
        right_arm_keypoints = get_distributed_points(right_arm_points, 3)
        left_leg_keypoints = get_distributed_points(left_leg_points, 3)
        right_leg_keypoints = get_distributed_points(right_leg_points, 3)
        
        # Combine all keypoints
        all_keypoints = spine_keypoints + left_arm_keypoints + right_arm_keypoints + left_leg_keypoints + right_leg_keypoints
        
        # If we don't have enough points from anatomical grouping, use K-means as fallback
        if len(all_keypoints) < 15:
            print("Not enough anatomical points found, using K-means fallback.")
            kmeans = KMeans(n_clusters=k, random_state=0).fit(yx)
            centers = kmeans.cluster_centers_.astype(int)
            
            # Try to organize K-means centers into anatomical regions
            centers_list = centers.tolist()
            centers_list.sort(key=lambda p: p[0])  # Sort by y-coordinate
            
            # Assign to anatomical regions based on sorted order
            spine_indices = [0, len(centers_list)//3, 2*len(centers_list)//3]  # Head, mid spine, lower spine
            left_arm_indices = [1, 2, 3]  # Shoulder, elbow, wrist
            right_arm_indices = [4, 5, 6]  # Shoulder, elbow, wrist
            left_leg_indices = [7, 8, 9]  # Hip, knee, ankle
            right_leg_indices = [10, 11, 12]  # Hip, knee, ankle
            
            # Final anatomical keypoints from K-means
            all_keypoints = [centers_list[i] for i in range(min(k, len(centers_list)))]
        
        # Draw keypoints with different colors for each body part
        keypoint_labels = []
        
        # Create a list of colors and point types for each body part
        body_parts = [
            ("Spine", all_keypoints[:3], (0, 0, 255)),       # Red - spine
            ("Left Arm", all_keypoints[3:6], (0, 255, 0)),   # Green - left arm
            ("Right Arm", all_keypoints[6:9], (255, 0, 0)),  # Blue - right arm
            ("Left Leg", all_keypoints[9:12], (0, 255, 255)),# Yellow - left leg
            ("Right Leg", all_keypoints[12:15], (255, 0, 255))# Magenta - right leg
        ]
        
        # Draw all keypoints
        for part_name, points, color in body_parts:
            for i, (y, x) in enumerate(points):
                if i == 0:
                    suffix = " (upper)"
                elif i == 1:
                    suffix = " (middle)"
                else:
                    suffix = " (lower)"
                
                label = part_name + suffix
                keypoint_labels.append(label)
                
                # Draw a colored circle at the keypoint position
                cv2.circle(skeleton_img, (x, y), 4, color, -1)
        
        # Define connections between keypoints to form a stick figure
        connections = []
        
        # Only add connections if we have at least 15 points
        if len(all_keypoints) >= 15:
            # Spine connections (0-1-2)
            connections.append((all_keypoints[0], all_keypoints[1]))  # Spine top to middle
            connections.append((all_keypoints[1], all_keypoints[2]))  # Spine middle to bottom
            
            # Left arm (3-4-5)
            connections.append((all_keypoints[1], all_keypoints[3]))  # Spine mid to left shoulder
            connections.append((all_keypoints[3], all_keypoints[4]))  # Left shoulder to elbow
            connections.append((all_keypoints[4], all_keypoints[5]))  # Left elbow to wrist
            
            # Right arm (6-7-8)
            connections.append((all_keypoints[1], all_keypoints[6]))  # Spine mid to right shoulder
            connections.append((all_keypoints[6], all_keypoints[7]))  # Right shoulder to elbow
            connections.append((all_keypoints[7], all_keypoints[8]))  # Right elbow to wrist
            
            # Left leg (9-10-11)
            connections.append((all_keypoints[2], all_keypoints[9]))  # Spine bottom to left hip
            connections.append((all_keypoints[9], all_keypoints[10]))  # Left hip to knee
            connections.append((all_keypoints[10], all_keypoints[11]))  # Left knee to ankle
            
            # Right leg (12-13-14)
            connections.append((all_keypoints[2], all_keypoints[12]))  # Spine bottom to right hip
            connections.append((all_keypoints[12], all_keypoints[13]))  # Right hip to knee
            connections.append((all_keypoints[13], all_keypoints[14]))  # Right knee to ankle
        
        # Draw connections
        for point1, point2 in connections:
            pt1 = (int(point1[1]), int(point1[0]))  # Convert to (x, y) for cv2.line
            pt2 = (int(point2[1]), int(point2[0]))
            cv2.line(skeleton_img, pt1, pt2, (255, 255, 255), 2)  # White lines
        
        # Return the image with skeleton drawn
        return cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2RGB) 