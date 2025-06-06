# Human Skeleton Estimation

This application estimates human skeleton from images using classical computer vision and pattern recognition techniques without deep learning.

## Features

- Real-time camera feed
- Background subtraction for silhouette extraction
- Step-by-step image processing visualization
- Human skeleton estimation using classical CV techniques

## Requirements

- Python 3.7+
- Webcam

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python human_skeleton_estimation.py
   ```

2. Follow these steps in the application:
   - Click "Capture Background" to capture a static background image
   - Position yourself in front of the camera and click "Capture Person"
   - Click "Process & Estimate Skeleton" to start the skeleton estimation process
   - View the results in the tabbed interface on the right panel

## Image Processing Steps

1. Negation - Inverting the image to highlight the subject
2. Threshold (2*mean) - Applying threshold to separate subject from background
3. Silhouette Extraction - Extracting the human silhouette
4. Silhouette Negation - Inverting the silhouette
5. Gaussian Smoothing - Smoothing the silhouette with Gaussian filter
6. Histogram Threshold - Applying threshold based on histogram analysis
7. Thinning - Thinning the silhouette
8. Skeletonization - Creating a 1-pixel wide skeleton
9. Interest Points Detection - Finding joints and endpoints
10. Final Skeleton - Visualizing the estimated human skeleton #   D I P R  
 