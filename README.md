# building an autonomous path planning for drone intelligence:

This AI application is built for autonomous path planning process using computer vision.

1. How it works:
The live footage is being gathered as visual input data from Drone camera. Then the application will initialze the algorithm that goes through every frame within miliseconds. algorithm will detect obstacles on the frame and mark them as dont go area. Then the algorithm will find the cleared areas in the frame where there is no obstacle and identify them as detected paths. 

2. implementation:

* Detects obstacles using edge detection and intensity thresholding
* Marks obstacles in red in the visualization
* Identifies clear areas where there are no obstacles
* Marks clear paths in green and labels them as "PATH   DETECTED"

3. How to use the application codebase:

* Make sure you have the required libraries installed (opencv-python and numpy)
* Connect your drone camera to your computer, or use your computer's webcam for testing
* Run the script, which will process the camera feed in real-time

There are two application code-bases.
  1. live_detector.py : for real-time detection using drone camera (alternatively webcam)

  2. test_detector.py : for testing the program with test footage/video 

4. Future improvements:
* Add depth perception: Integrate depth information if your drone has a stereo camera or depth sensor

* Implement machine learning: Train a model to better identify different types of obstacles

* Add path optimization: Calculate the optimal path through the detected clear areas

* Drone control integration: Connect this path planning to your drone's flight controller



