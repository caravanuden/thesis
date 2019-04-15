#!/bin/bash
echo "docker Dockerfile
singularity Singularity" | while read backend filename; do
    docker run --rm kaczmarj/neurodocker:master generate $backend  \
      -b centos -p yum --copy docker_r21d_dependencies.sh / \
      -r /docker_r21d_dependencies.sh \
      -e PYTHONPATH=/usr/local/lib/python2.7/site-packages/  LD_LIBRARY_PATH=/usr/local/lib PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
      | sed -e 's,/tmp/\* .*,,g' >| $filename
done
