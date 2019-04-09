#!/bin/bash
# installs dependencies for Res(2+1)D - OpenCV, ffmpeg, and Caffe2 based on FB VMZ installation guide
# used in Dockerfile in /ihome/cara/thesis
# TODO:
#		use e.g. /tmp/build as the directory where you clone for building etc
# CVU 2019

set -eu

# Portions which are known to be good are in _good*

cd /tmp/build
# get opencv
wget https://github.com/opencv/opencv/archive/3.4.0.zip -O opencv-3.4.0.zip
unzip opencv-3.4.0.zip
cd opencv-3.4.0
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D BUILD_EXAMPLES=ON \
	-D BUILD_SHARED_LIBS=ON ..
make -j8
make install
ldconfig


# H.264 video encoder
cd /tmp/build
git clone http://git.videolan.org/git/x264.git
cd x264
./configure --enable-shared --enable-pic
make -j8
make install

# ogg bitstream library
cd /tmp/build
curl -O -L http://downloads.xiph.org/releases/ogg/libogg-1.3.3.tar.gz
tar xzvf libogg-1.3.3.tar.gz
cd libogg-1.3.3
./configure
make -j8
make install

# ffmpeg
cd /tmp/build
git clone http://git.videolan.org/git/ffmpeg.git
cd ffmpeg
 ./configure --enable-gpl --enable-nonfree --enable-libtheora --enable-libvorbis  --enable-libx264  --enable-postproc --enable-version3 --enable-pthreads --enable-shared --enable-pic
make -j8
make install

# make sure pkg config and the linker can see ffmpeg
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
hash -r
ldconfig

# get caffe2
yum install -y protobuf-devel leveldb-devel snappy-devel opencv-devel lmdb-devel python-devel gflags-devel glog-devel kernel-devel

# get cuDNN
cd /tmp/build
cp cuda/lib64/* /usr/local/cuda/lib64/
cp cuda/include/cudnn.h /usr/local/cuda/include/
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# make sure nvcc is runnable
nvcc --version

# python dependencies for caffe2
pip install lmdb numpy flask future graphviz hypothesis jupyter matplotlib protobuf pydot python-nvd3 pyyaml requests scikit-image scipy six tornado

# build caffe2
cd /tmp/build
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch && git submodule update --init

# modify CMakeLists.txt to make USE_FFMPEG ON
# after cmake (see below), check the output log, makesure USE_OPENCV: ON and USE_FFMPEG: ON
mkdir build
cd build
cmake ..
sudo make -j8 install

export PYTHONPATH=$PYTHONPATH:/usr/local/pytorch

# clean up
rm -rf /tmp/build
