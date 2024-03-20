FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt -y update && apt -y upgrade && apt -y install git cmake build-essential lsb-release sudo && rm -rf /var/lib/apt/lists

WORKDIR /app

RUN apt -y update && \
    apt install -y --no-install-recommends build-essential cmake doxygen \
        g++ git octave python-dev python-setuptools wget mlocate \
        python2 curl qt5-default minizip \
        ann-tools libann-dev            \
        libassimp-dev libavcodec-dev libavformat-dev libeigen3-dev libfaac-dev          \
        libflann-dev libfreetype6-dev liblapack-dev libglew-dev libgsm1-dev             \
        libmpfi-dev  libmpfr-dev liboctave-dev libode-dev libogg-dev libpcre3-dev       \
        libqhull-dev libswscale-dev libtinyxml-dev libvorbis-dev libx264-dev            \
        libxml2-dev libxvidcore-dev libbz2-dev \
        libsoqt520-dev \
        libccd-dev                  \
        libcollada-dom2.4-dp-dev liblog4cxx-dev libminizip-dev octomap-tools \
        libboost-all-dev libboost-python-dev && \
    rm -rf /var/lib/apt/lists

RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py && \
    python2 get-pip.py && \
    pip install ipython h5py numpy scipy wheel && \
    rm get-pip.py

RUN mkdir -p ~/git  && \
    cd ~/git && git clone https://github.com/Tencent/rapidjson.git && \
    cd rapidjson && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j `nproc` && make install && \
    cd ../ && rm build -rf

# Install Pybind
RUN git config --global user.email "docker@example.com"
RUN git config --global user.name "docker"
RUN cd ~/git && git clone https://github.com/pybind/pybind11.git  && \
    cd pybind11 && mkdir build && cd build  && \
    git remote add woody https://github.com/woodychow/pybind11.git \
        && git fetch woody && git checkout v2.2.4 \
        && git cherry-pick 94824d68a037d99253b92a5b260bb04907c42355 \
        && git cherry-pick 98c9f77e5481af4cbc7eb092e1866151461e3508 \
        && cmake .. -DPYBIND11_TEST=OFF -DPythonLibsNew_FIND_VERSION=2 \
        && make install

ENV OSG_COMMIT 1f89e6eb1087add6cd9c743ab07a5bce53b2f480  
RUN mkdir -p ~/git && cd ~/git && \
    git clone https://github.com/openscenegraph/OpenSceneGraph.git && \
    cd OpenSceneGraph && git reset --hard ${OSG_COMMIT} && \
    mkdir build && cd build && \
    cmake -DDESIRED_QT_VERSION=4 .. -DCMAKE_BUILD_TYPE=Release && \
    make -j `nproc` &&\
    make install &&\
    make install_ld_conf && \
    cd .. && rm build -rf


RUN mkdir -p ~/git&& cd ~/git &&\
    git clone https://github.com/flexible-collision-library/fcl &&\
    cd fcl&& git reset --hard 0.5.0 &&\
    mkdir build&& cd build &&\
    cmake .. -DCMAKE_BUILD_TYPE=Release &&\
    make -j `nproc` &&\
    make install &&\
    cd .. && rm build -rf


RUN pip install --upgrade --user sympy==0.7.1
ENV RAVE_COMMIT 2024b03554c8dd0e82ec1c48ae1eb6ed37d0aa6e
RUN mkdir -p ~/git && cd ~/git &&\
	git clone -b production https://github.com/rdiankov/openrave.git &&\
    cd openrave && git reset --hard ${RAVE_COMMIT} &&\
    mkdir build && cd build &&\
  	cmake -DODE_USE_MULTITHREAD=ON -DOSG_DIR=/usr/local/lib64/ \
  		-DUSE_PYBIND11_PYTHON_BINDINGS:BOOL=TRUE 			   \
  		-DBoost_NO_BOOST_CMAKE=1 .. \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
    make -j $(nproc) && \
    make install
RUN pip install pyopengl
CMD ["openrave.py"]
