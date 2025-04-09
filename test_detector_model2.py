# tet model 2 for testing footage:
import cv2
import numpy as np
import time
import argparse
import sys

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
        print(f"Opening video file: {video_path}")
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Verify parameters are valid
        if obstacle_threshold is None:
            obstacle_threshold = 0.4
            print("Warning: obstacle_threshold was None, using default value 0.4")
        
        if min_path_width is None:
            min_path_width = 50
            print("Warning: min_path_width was None, using default value 50")
            
        self.obstacle_threshold = float(obstacle_threshold)
        self.min_path_width = int(min_path_width)
        self.output_path = output_path
        self.video_writer = None
        
        # Print video properties for debugging
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video dimensions: {self.frame_width}x{self.frame_height}")
        print(f"Video FPS: {self.fps}")
        print(f"Frame count: {self.frame_count}")
        print(f"Obstacle threshold: {self.obstacle_threshold}")
        print(f"Minimum path width: {self.min_path_width}")
        
    def detect_obstacles(self, frame):
        """
        Detect obstacles in the frame using edge detection and thresholding
        
        Args:
            frame: Input camera frame
            
        Returns:
            Mask where obstacles are white (255) and free space is black (0)
        """
        try:
            # Check if frame is valid
            if frame is None or frame.size == 0:
                print("Error: Empty or invalid frame")
                return np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
            
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
            threshold_value = int(255 * self.obstacle_threshold)
            _, threshold = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
            
            # Combine edge detection and threshold results
            obstacle_mask = cv2.bitwise_or(dilated_edges, threshold)
            
            # Apply morphological operations to clean up the mask
            obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)
            
            return obstacle_mask
            
        except Exception as e:
            print(f"Error in detect_obstacles: {e}")
            print(f"Frame shape: {frame.shape if frame is not None else 'None'}")
            print(f"Frame type: {type(frame)}")
            return np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
    
    def find_clear_paths(self, obstacle_mask):
        """
        Find clear paths in the frame based on the obstacle mask
        
        Args:
            obstacle_mask: Binary mask where obstacles are white (255)
            
        Returns:
            List of path regions (x, y, width, height)
        """
        try:
            # Check if mask is valid
            if obstacle_mask is None or obstacle_mask.size == 0:
                print("Error: Empty or invalid obstacle mask")
                return []
                
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
            
        except Exception as e:
            print(f"Error in find_clear_paths: {e}")
            return []
    
    def visualize_results(self, frame, obstacle_mask, paths):
        """
        Visualize the detected obstacles and clear paths
        
        Args:
            frame: Original camera frame
            obstacle_mask: Binary mask where obstacles are white (255)
            paths: List of path regions (x, y, width, height)
            
        Returns:
            Visualization frame with marked obstacles and paths
        """
        try:
            # Create a copy of the original frame
            result = frame.copy()
            
            # Ensure obstacle_mask has correct dimensions
            if len(obstacle_mask.shape) == 2:  # Single channel
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
            
        except Exception as e:
            print(f"Error in visualize_results: {e}")
            return frame  # Return original frame if visualization fails
    
    def process_frame(self, frame):
        """
        Process a single frame for obstacle detection and path planning
        
        Args:
            frame: Input camera frame
            
        Returns:
            Processed frame with visualizations
        """
        try:
            # Detect obstacles
            obstacle_mask = self.detect_obstacles(frame)
            
            # Find clear paths
            paths = self.find_clear_paths(obstacle_mask)
            
            # Visualize results
            result = self.visualize_results(frame, obstacle_mask, paths)
            
            return result
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return frame  # Return original frame if processing fails
    
    def initialize_video_writer(self, frame_width, frame_height):
        """Initialize video writer for saving processed video"""
        if self.output_path:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(
                    self.output_path, fourcc, self.fps, (frame_width, frame_height)
                )
                print(f"Video writer initialized for output: {self.output_path}")
            except Exception as e:
                print(f"Error initializing video writer: {e}")
                self.output_path = None
    
    def run(self):
        """Run the path planning system on the test video"""
        try:
            frame_count = 0
            processing_times = []
            
            while True:
                # Capture frame from video
                ret, frame = self.cap.read()
                if not ret:
                    print(f"End of video or error reading frame at frame {frame_count}")
                    break
                
                frame = cv2.resize(frame, (640, 480)) # resizing the output frame
                # Initialize video writer on first frame
                if frame_count == 0 and self.output_path:
                    self.initialize_video_writer(frame.shape[1], frame.shape[0])
                
                start_time = time.time()
                
                # Process the frame
                result = self.process_frame(frame)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                fps = 1.0 / processing_time if processing_time > 0 else 0
                
                # Add frame number and FPS to the result
                cv2.putText(result, f"Frame: {frame_count} | FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Save processed frame if output path is specified
                if self.video_writer and self.video_writer.isOpened():
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
                
                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames")
            
            # Print performance statistics
            if processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                avg_fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"\nProcessed {frame_count} frames")
                print(f"Average processing time: {avg_time:.4f} seconds per frame")
                print(f"Average FPS: {avg_fps:.2f}")
                
        except Exception as e:
            print(f"Error in run method: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            if hasattr(self, 'video_writer') and self.video_writer is not None:
                self.video_writer.release()
            cv2.destroyAllWindows()
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()


def main():
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
        # Print arguments for debugging
        print("Running with parameters:")
        print(f"  Video: {args.video}")
        print(f"  Threshold: {args.threshold}")
        print(f"  Min Width: {args.min_width}")
        print(f"  Output: {args.output}")
        
        # Create and run the path planner with test video
        planner = DronePathPlanner(
            video_path=args.video,
            obstacle_threshold=args.threshold,
            min_path_width=args.min_width,
            output_path=args.output
        )
        
        planner.run()
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())