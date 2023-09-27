
# This script will install wiki-llm on Linux VMs that are created using Azure Marketplace's pytorch image

# Install Anaconda3
# curl -o Anaconda-latest-Linux-x86_64.sh https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
# bash Anaconda-latest-Linux-x86_64.sh
# rm Anaconda-latest-Linux-x86_64.sh

# Create the conda environment
# conda env create --file conda_env.yml
# conda activate wiki_llm
# python -m spacy download en_core_web_sm

# Download azcopy
# wget https://aka.ms/downloadazcopy-v10-linux
# tar -xvf downloadazcopy-v10-linux
# rm downloadazcopy-v10-linux
# mkdir ~/bin
# mv ./azcopy_linux_amd64_*/azcopy ~/bin/
# rm -r ./azcopy_linux_amd64_*/
# export PATH="$PATH:~/bin/"
# echo 'export PATH=$PATH:~/bin/' >> ~/.bashrc

# Download ColBERT model
# wget https://almond-static.stanford.edu/research/wiki_llm/colbert_model.tar.gz
# tar -xzf colbert_model.tar.gz
# rm colbert_model.tar.gz

# Download ColBERT index
# azcopy copy https://nfs009a5d03c43b4e7e8ec2.blob.core.windows.net/genie-public/research/wiki_llm/colbert_wikipedia_index_04-28-2023.tar ./
# tar -xvf colbert_wikipedia_index_04-28-2023.tar
# rm colbert_wikipedia_index_04-28-2023.tar