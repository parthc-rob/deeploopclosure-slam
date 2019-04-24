# DeepLCD

This folder contains a modified version of DeepLCD library:

`<https://github.com/rpng/calc/tree/master/DeepLCD>`

A C++ library for online SLAM loop closure, using  models. Included with the shared library is one pre-trained model (see get_model.sh), which is downloaded on compilation, a useful demo, unit testing (CPU tests only), and a speed test, as well as an online loop closure demo with ROS!

## Dependencies

Required:

- OpenCV >= 2.0
- Eigen >= 3.0
- Boost filesystem
- Caffe

## Model

To get the pre-trained model, please run `get_model.sh`



## Notice

Dear instructor : start an issue in our repo if there is anything wrong with the installationl!!! We would be very glad to help you deal with any issues you may have when running it (we have gone through a LOT of installation and compile errors).



## TO Install Caffe

Look at this repo for detailed tutorial `<https://gist.github.com/nikitametha/c54e1abecff7ab53896270509da80215>`

We don't have GPU so only install and use it under CPU mode.

A customed setting for Caffe `makefile.config`  has been uploaded.  Please use this config and put it in the root folder of `caffe`and it will save you a lot of time. 

If you meet any unexpected error after using our config file, search over google and most of the errors can be fixed by installing some dependencies by doing a simple `pip install <package name> `

## To Compile

```
$ mkdir build && cd build
$ cmake .. && make # Already set to Release build
```

Note that if your caffe is not installed in ~/caffe, you must use

```
$ cmake -DCaffe_ROOT_DIR=</path/to/caffe> .. && make
```

instead, otherwise you will see the error 

`No such file: caffe.h`

## To Run Tests

```
$ cd /path/to/build
$ ./deeplcd-test
```

## To Run the Demo

The demo uses 6 images that are included in this repo. They are from the [KITTI VO dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). They are stereo pairs located in src/images/live and src/images/memory. Each image in live/memory with the same file name is from the same point in time. The demo first loads the memory images into a database, then performs a search with the live images. The default model downloaded by get_model.sh is used here, and caffe is set to cpu. You should see matching image IDs in the printed output. That means that the search successfully matched the memory images to the live ones. 

## To Run the Speed Test

```
$ speed-test <mem dir> <live dir> <(optional) GPU_ID (default=-1 for cpu)>
```

where mem/live dir are directories containing images to use in the test. For example, if you have two directories for left/right stereo pairs, you can throw those in the arguments. GPU_ID defaults to -1, which means use the CPU.

## TO Run the Vary Database Size Test

```$ vary-db-size <mem dir> <live dir> <(optional) GPU_ID (default=-1 for cpu)```

This will test the query speed for different database size and save the result to `vary-db-size-calc-results.txt`

## Online Loop Closure Demo with ROS

See online-demo-ws

