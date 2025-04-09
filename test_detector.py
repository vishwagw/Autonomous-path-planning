# for testing with test footages/videos:

import cv2
import numpy as np
import time
import argparse

# building the dronepath planner class:
class DronePathPlanner:
    def __init__(self, video_path, obstacle_threshold=0.4, min_path_width=50, output_path=None):
        """
        Initialize the drone path planner
        
        Args:
            video_path: Path to the test video file
            obstacle_threshold: Threshold for obstacle detection (lower values detect more obstacles)
            min_path_width: Minimum width in pixels for a valid path
            output_path: Path to save the processed video (None for no saving)
        """
        video_path = './in1.mp4'
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        self.obstacle_threshold = obstacle_threshold
        self.min_path_width = min_path_width
        self.output_path = output_path
        self.video_writer = None
    
    # detecting obstacles:
    def detect_obstacles(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to make obstacles more prominent
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Additional processing based on intensity (darker areas may be obstacles)
        _, threshold = cv2.threshold(gray, int(255 * self.obstacle_threshold), 255, cv2.THRESH_BINARY_INV)
        
        # Combine edge detection and threshold results
        obstacle_mask = cv2.bitwise_or(dilated_edges, threshold)
        
        # Apply morphological operations to clean up the mask
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)

    #  Find clear paths in the frame based on the obstacle mask
    def find_clear_paths(self, obstacle_mask):
         # Invert obstacle mask to get free space
        free_space = cv2.bitwise_not(obstacle_mask)
        
        # Find contours in the free space
        contours, _ = cv2.findContours(free_space, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        paths = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= self.min_path_width:
                paths.append((x, y, w, h))
        
        return paths
    
    #  Visualize the detected obstacles and clear paths
    def visualize_results(self, frame, obstacle_mask, paths):
        # Create a copy of the original frame
        result = frame.copy()
        
        # Mark obstacles in red (using the obstacle mask)
        red_mask = np.zeros_like(frame)
        red_mask[:, :, 2] = obstacle_mask  # Red channel
        result = cv2.addWeighted(result, 1, red_mask, 0.5, 0)
        
        # Mark clear paths in green
        for path in paths:
            x, y, w, h = path
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, "PATH DETECTED", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result
    
    #  Process a single frame for obstacle detection and path planning
    def process_frame(self, frame):
        # Detect obstacles
        obstacle_mask = self.detect_obstacles(frame)
        
        # Find clear paths
        paths = self.find_clear_paths(obstacle_mask)
        
        # Visualize results
        result = self.visualize_results(frame, obstacle_mask, paths)
        
        return result
    
    # Initialize video writer for saving processed video
    def initialize_video_writer(self, frame_width, frame_height):
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, fps, (frame_width, frame_height)
            )
    
    # Run the path planning system on the test video
    def run(self):
        try:
            frame_count = 0
            processing_times = []
            
            while True:
                # Capture frame from video
                ret, frame = self.cap.read()
                if not ret:
                    print(f"End of video or error reading frame at frame {frame_count}")
                    break
                
                frame = cv2.resize(frame, (640, 480))  # Resize for consistency
                # Initialize video writer on first frame
                if frame_count == 0 and self.output_path:
                    self.initialize_video_writer(frame.shape[1], frame.shape[0])
                
                start_time = time.time()
                
                # Process the frame
                result = self.process_frame(frame)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                fps = 1.0 / processing_time
                
                # Add frame number and FPS to the result
                cv2.putText(result, f"Frame: {frame_count} | FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Save processed frame if output path is specified
                if self.video_writer:
                    self.video_writer.write(result)
                
                # Display the result
                cv2.imshow("Drone Cam: 001", result)
                
                # Exit on 'q' key press or ESC
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 27 is ESC
                    break
                
                # Pause on spacebar
                if key == 32:  # 32 is spacebar
                    cv2.waitKey(0)
                
                frame_count += 1
            
            # Print performance statistics
            if processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                avg_fps = 1.0 / avg_time
                print(f"\nProcessed {frame_count} frames")
                print(f"Average processing time: {avg_time:.4f} seconds per frame")
                print(f"Average FPS: {avg_fps:.2f}")
                
        finally:
            self.cap.release()
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()

    # cleaning up resources
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()

# initializing the test program:
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Drone Path Planning with Test Video')
    parser.add_argument('--video', type=str, required=True, 
                        help='Path to test video file')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Obstacle detection threshold (default: 0.4)')
    parser.add_argument('--min-width', type=int, default=50,
                        help='Minimum path width in pixels (default: 50)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save processed video (optional)')
    
    args = parser.parse_args()

    try:
        # Create and run the path planner with test video
        planner = DronePathPlanner(
            video_path=args.video,
            obstacle_threshold=args.threshold,
            min_path_width=args.min_width,
            output_path=args.output
        )
        
        planner.run()
        
    except Exception as e:
        print(f"Error: {e}")

