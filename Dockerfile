# Dev Test:
# Build container with `docker build --tag im_backend_dev:latest`
# Run container with `docker run -it im_backend_dev bash`
FROM ubuntu:20.04

WORKDIR /im_backend_dev

ENV TZ=US \
    DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install git -y
RUN apt -y install cmake build-essential qt5-default libeigen3-dev libsuitesparse-dev python3-dev pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# COPY setup_g2opy.sh setup_g2opy.sh
# RUN ./setup_g2opy.sh
RUN git clone https://github.com/occamLab/g2opy
RUN cd g2opy && mkdir build && cd build && cmake .. && make -j8 && make install && cd .. && python3 setup.py install

COPY . .

CMD ["bash"]