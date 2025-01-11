#!/bin/bash
# Install PyTorch and Python Packages

# conda create -n dmcg python=3.11 -y
# conda activate dmcg

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y 

pip install protobuf==3.20.* sacred numpy scipy gym==0.11 matplotlib seaborn \
    pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger pymongo setproctitle 
pip install git+https://github.com/oxwhirl/smacv2.git 

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html 
pip install pogema pogema-toolbox 