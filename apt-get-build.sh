#!/usr/bin/env bash

apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libpng-dev \
    libjpeg-dev \
    libjxl-dev
