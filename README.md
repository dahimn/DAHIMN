# Introduction
The Code for the paper "Incorporating Dual-Aware with Hierarchical Interactive Memory Networks for Task-Oriented Dialogue".

# Requirements

CUDA version：11.2

Python3.7.4


    pip install -r requirements.txt
    chomd 777 utils/multi-bleu.perl



# Dataset
The In-Car Assistant (KVR) dataset and CamRest dataset are all in the "data" folder

# Training and Testing  

**For training:** 

For In-Car Assistant (KVR) dataset

	CUDA_VISIBLE_DEVICES=0 python myTrain.py -lr=0.001 -hop=3 -l=2 -hdd=128 -dr=0.3 -dec=DAHIMN -bsz=32 -ds=kvr -tfr=0.8 -es=ENTF1

For CamRest dataset

	CUDA_VISIBLE_DEVICES=0 python myTrain.py -lr=0.001 -hop=3 -l=2 -hdd=128 -dr=0.3 -dec=DAHIMN -bsz=32 -ds=cam -tfr=0.8 -es=ENTF1

 

**For testing the reported results:** 

For In-Car Assistant (KVR) dataset

    CUDA_VISIBLE_DEVICES=0 python myTest.py -ds=kvr -path=reportSave/DAHIMN-KVR/HDD128BSZ32DR0.3HOP3L2lr0.001tfr0.8ENTF1 -rec=1 

For CamRest dataset

    CUDA_VISIBLE_DEVICES=0 python myTest.py -ds=cam -path=reportSave/DAHIMN-CAM/HDD128BSZ32DR0.3HOP3L2lr0.001tfr0.8ENTF1 -rec=1



