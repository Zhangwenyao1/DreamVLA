apt-get update
# Install dependency for calvin
apt-get -yqq install libegl1 libgl1

# Install EGL mesa
apt-get install -yqq libegl1-mesa libegl1-mesa-dev
apt-get install -yqq mesa-utils libosmesa6-dev llvm
apt-get -yqq install meson
apt-get -yqq build-dep mesa
