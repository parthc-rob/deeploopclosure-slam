## Online Ros Demo

## Dependencies

- deep-lcd compiled as directed in the parent directory's README
- ROS (tested on Kinetic) 

## Build

Run the catkin build script here, or just use `catkin_make`. The `CMakeLists.txt` already has it set to release mode.

Note that if your caffe is not installed in ~/caffe, you must use

```$ catkin_make -DCaffe_ROOT_DIR=</path/to/caffe> ```

## Data

We use the kitti2bag tool to convert KITTI sequence to a rosbag.`<https://github.com/tomas789/kitti2bag>`

## Run

1. You will need to have roscore running first. 

2. This demo requires an image topic and a TransformStamped topic and we have developed a tool in python to publish these topics.  Keep this program running in the background: 

   `python kitti_tf_publisher.py`

3. Then Run with:

```
$ source devel/setup.bash
$ roslaunch launch/online-demo.launch
```

​		Rviz should open, and you should see many lines of caffe logging output. 

4. After that, you need to play the rosbag file to start publishing messages. 

   `rosbag play <yourdata.bag>`

5. Then you should see the demo as we shown here

   `https://youtu.be/uQfligCUiIU`

