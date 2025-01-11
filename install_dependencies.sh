#!/bin/bash
# Install PyTorch and Python Packages

# conda create -n dmcg python=3.11 -y
# conda activate dmcg

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y 
pip install git+https://github.com/oxwhirl/smacv2.git 
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html # Optional dependencies: 
pip install pymongo setproctitle sacred pyyaml tensorboard_logger matplotlib 