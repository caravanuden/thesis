# Generated by Neurodocker version 0.4.1-19-gf52d03d
# Timestamp: 2019-04-15 17:31:00 UTC
#
# Thank you for using Neurodocker. If you discover any issues
# or ways to improve this software, please submit an issue or
# pull request on our GitHub repository:
#
#     https://github.com/kaczmarj/neurodocker

Bootstrap: docker
From: centos

%post
export ND_ENTRYPOINT="/neurodocker/startup.sh"
yum install -y -q \
    bzip2 \
    ca-certificates \
    curl \
    localedef \
    unzip
yum clean packages
rm -rf /var/cache/yum/*
localedef -i en_US -f UTF-8 en_US.UTF-8
chmod 777 /opt && chmod a+s /opt
mkdir -p /neurodocker
if [ ! -f "$ND_ENTRYPOINT" ]; then
  echo '#!/usr/bin/env bash' >> "$ND_ENTRYPOINT"
  echo 'set -e' >> "$ND_ENTRYPOINT"
  echo 'if [ -n "$1" ]; then "$@"; else /usr/bin/env bash; fi' >> "$ND_ENTRYPOINT";
fi
chmod -R 777 /neurodocker && chmod a+s /neurodocker

/thesis_dependencies.sh

echo '{
\n  "pkg_manager": "yum",
\n  "instructions": [
\n    [
\n      "base",
\n      "centos"
\n    ],
\n    [
\n      "_header",
\n      {
\n        "version": "generic",
\n        "method": "custom"
\n      }
\n    ],
\n    [
\n      "copy",
\n      [
\n        "thesis_dependencies.sh",
\n        "/"
\n      ]
\n    ],
\n    [
\n      "run",
\n      "/thesis_dependencies.sh"
\n    ],
\n    [
\n      "env",
\n      {
\n        "PYTHONPATH": "/usr/local/lib/python2.7/site-packages/",
\n        "LD_LIBRARY_PATH": "/usr/local/lib",
\n        "PKG_CONFIG_PATH": "/usr/local/lib/pkgconfig"
\n      }
\n    ]
\n  ]
\n}' > /neurodocker/neurodocker_specs.json

%environment
export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export ND_ENTRYPOINT="/neurodocker/startup.sh"
export PYTHONPATH="/usr/local/lib/python2.7/site-packages/"
export LD_LIBRARY_PATH="/usr/local/lib"
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig"

%files
thesis_dependencies.sh /

%runscript
/neurodocker/startup.sh "$@"
