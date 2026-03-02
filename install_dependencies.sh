#!/bin/bash
set -e

if [ ! -f "install_dependencies.sh" ]; then
    echo "Please run this script in the root folder of the repo."
    exit 1
fi

# useful vars for later
verl_commit=15263cb86a464264edb1e5462675e25ddf6ff9d8
proj_root=$(pwd)

# push some environment variables
echo "ROOT=$(pwd)" > .env

# create and activate conda env
conda create -n rlcausal python==3.9 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rlcausal

# get verl and patch
if [ -e "lib/verl" ]; then
    echo "Verl already downloaded, skipping clone."
else
    git clone https://github.com/volcengine/verl.git lib/verl
fi
cd lib/verl
git checkout $verl_commit
pip3 install -e . --no-cache-dir
git apply ../patches/250330.patch
git apply ../patches/250510.patch
cd $proj_root

# additional dependencies
pip install addict --no-cache-dir
pip install scipy --no-cache-dir
pip install pydot==3.0.4 --no-cache-dir
pip install vllm==0.6.3 --no-cache-dir
pip install transformers==4.51.3
pip install aiohttp==3.11.18 --no-cache-dir
pip install ray==2.46.0 --no-cache-dir
pip install flash-attn==2.7.4.post1 --no-build-isolation
