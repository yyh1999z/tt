conda create -n attack python=3.7.15

pip install wget

pip install numpy==1.21.6

pip install pandas

pip install tqdm

pip install libarchive-c

pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install tensorboard


运行命令：
训练模型
python train_robust.py 

攻击模型
python attack.py
