# FEUP-RVA
# Download OpenCV source

$ wget https://github.com/opencv/opencv/archive/3.4.16.zip

# Install Dependencies:

$ sudo apt-get install libavformat-dev libavutil-dev libswscale-dev

$ sudo apt-get install libv4l-dev

$ sudo apt-get install libglew-dev

$ sudo apt-get install libgtk2.0-dev

$ sudo apt-get install python-opencv

After installing the dependencies, now we need to build and install OpenCV using the following commands:

$ unzip opencv-3.4.16.zip

$ cd opencv-3.4.16

$ mkdir build && cd build

$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_TBB=OFF -D BUILD_TBB=OFF -D WITH_V4L=ON -D WITH_LIBV4L=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF ..

Compile and install:

The following command will build and install OpenCV libraries in the location – “/usr/local/lib/”

$ sudo make install

Now compile and run the test program:

$ python3 test.py
