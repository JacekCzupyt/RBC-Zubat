#!/usr/bin/env bash

arch=$(dpkg --print-architecture)
curl -fSsL -O "https://github.com/NVIDIA/enroot/releases/download/v3.4.0/enroot_3.4.0-1_${arch}.deb"
sudo apt install -y ./*.deb
