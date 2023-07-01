FROM arm64v8/ubuntu:jammy

RUN apt-get update && \
    apt-get install -y  git python3 python3-pip libeigen3-dev cmake libsuitesparse-dev python3-dev vim  sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN git clone https://github.com/occamLab/invisible-map-generation

WORKDIR invisible-map-generation
RUN python3 -m pip install -r requirements.txt
RUN git clone https://github.com/occamLab/g2opy

# patch CMakeLists.txt to get rid of incorrect optimization flags
RUN sed -i.old -E '/msse/d' g2opy/CMakeLists.txt
WORKDIR g2opy
run mkdir build
WORKDIR build
RUN cmake ..
RUN make -j4
WORKDIR ..
RUN python3 setup.py install
COPY simple_install_test.py simple_install_test.py
CMD python3 simple_install_test.py
