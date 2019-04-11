# installs dependencies for Res(2+1)D - OpenCV, ffmpeg, and Caffe2
# based on FB VMZ installation guide
# CVU 2019
# Does not install CUDA atm due to licensing restrictions etc

# use the most recent centOS
FROM centos:latest

COPY docker_r21d_dependencies*.sh /tmp/
RUN bash /tmp/docker_r21d_dependencies.sh

# Set env vars so they are available in the generated image as well
ENV PYTHONPATH /usr/local/lib/python2.7/site-packages/
ENV LD_LIBRARY_PATH /usr/local/lib
ENV PKG_CONFIG_PATH /usr/local/lib/pkgconfig

# to confirm installed correctly
RUN python -c 'from caffe2.python import core'
