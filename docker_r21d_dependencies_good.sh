#!/bin/bash
# installs dependencies for Res(2+1)D - OpenCV, ffmpeg, and Caffe2 based on FB VMZ installation guide
# used in Dockerfile in /ihome/cara/thesis
# TODO:
#		use e.g. /tmp/build as the directory where you clone for building etc
# CVU 2019

set -eu

# get ffmpeg
yum install -y autoconf automake bzip2 freetype-devel gcc gcc-c++ git libtool pkgconfig zlib-devel yasm-devel libtheora-devel libvorbis-devel libX11-devel gtk2-devel

pip install pip setuptools -U

# get cmake, must be > 3.7
yum install -y cmake
cmake --version

mkdir /tmp/build
cd /tmp/build

