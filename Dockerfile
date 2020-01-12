FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y apt-utils \
                       python3-pip \
                       python3-dev \
                       pkg-config \
                       libigraph0v5 \
                       libigraph0-dev && \
    pip3 install --upgrade --no-cache jupyter \
                                      numpy \
                                      scipy \
                                      python-igraph \
                                      tqdm
