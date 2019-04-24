## Deep Loop Closure SLAM

ROB 530 Final Project, Team 2 - Ning Xu, Kun Sun, Poorani Ravindhiran, Tribhi Kathuria, Parth Chopra

 In this work, we evaluate a place recognition algorithm for loop closure detection, building upon the ORB-SLAM2 framework. We evaluate the performance of the proposed neural network generated HOG-like descriptors[termed Deep HOG] against ORB features with Bag-of-Words[BoW] representation and Histogram of Oriented Gradients[HOG] for performing place recognition and matching. The study seeks to improve on the loop closing algorithm in ORB-SLAM2. Loop Closure is an essential component in SLAM that helps to create consistent environment maps and robot  trajectories essential for long-term autonomy of mobile robots. Finally, we show that the global image DHoG descriptors perform better than BoW representation with ORB features and traditional HOG descriptors in terms of accuracy and query time when evaluated over test datasets and show promise for being used for loop closure in a real SLAM system. 

### DeepLCD

In this folder, we have modified the deep loop closure detection library (DeepLCD) and using our own neural network model to improve the test result. 
* `online-demo_ws`: 2D demo showing loop closing detection using ROS Rviz visualization tool
* `speed-test`: test the runtime used for computing one image DHoG descriptor
* `vary-db-size`: test the query runtime for different database size

### benchmark_code
In this folder, we have reimplement the unsupervised neural network using Keras. Some tools for result evaluation are also provided.
* `prec_recall_figure.py` run our trained model on the KITTI gray sequence and generate the confusion matrix and precision recall curve
* `sim_score_hist.py` run our trained model on the CampusLoopDataset and plot the similarity score histogram
* `train.py`, `move.py` and `psedoDatasetGen.py` are used in model training.

### Pre-integration with ORB-SLAM2
To see the pre-integration development on ORB-SLAM2 with our proposed DHoG descriptor, please:  
Do `chmod +x get_repos.sh`.
Run `./get_repos.sh`, and read individual readmes in the folders
