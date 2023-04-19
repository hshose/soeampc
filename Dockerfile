FROM ubuntu:22.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update && apt-get install -y apt-utils

# config dependencies install
ARG DEBIAN_FRONTEND=noninteractive

# install dependencies
# THIS WILL REVERT EVERYTHING TO PYTHON3
RUN apt-get update && apt-get install -y \
	apt-utils \
	autoconf \
	automake \
	libtool \
	curl \
	make \
	g++ \
	wget \
	unzip \
	maven \
    git \
    build-essential \
    cmake \
    python3-pip \
    python3-yaml \
	software-properties-common 


# install acados
RUN git clone --branch v0.1.9 https://github.com/acados/acados --recursive
WORKDIR /acados
RUN mkdir build && cd build
WORKDIR /acados/build
RUN cmake -DACADOS_WITH_QPOASES=ON -DACADOS_SILENT=ON ..
RUN make -j4 install
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/acados/lib"
ENV ACADOS_SOURCE_DIR="/acados"
RUN pip3 install -e /acados/interfaces/acados_template
WORKDIR /acados/bin
RUN wget -qq https://github.com/acados/tera_renderer/releases/download/v0.0.34/t_renderer-v0.0.34-linux && \
    mv t_renderer-v0.0.34-linux t_renderer && \
    chmod +x /acados/bin/t_renderer


# install soeampc package dependencies
COPY ./soeampc/requirements.txt /soeampc/soeampc/requirements.txt
RUN pip3 install -r /soeampc/soeampc/requirements.txt

# install example dependencies
COPY ./examples/requirements.txt  /soeampc/examples/requirements.txt 
RUN pip3 install -r /soeampc/examples/requirements.txt


COPY . /soeampc
WORKDIR /soeampc

# docker build -t soeampc

# docker run -it -v $(pwd):/soeampc 