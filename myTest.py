from utils.config import *
from models.DAHIMN import *

import torch
import numpy as np 
import random


directory = args['path'].split("/")
task = directory[2].split('HDD')[0]
HDD = directory[2].split('HDD')[1].split('BSZ')[0]
HOP = directory[2].split('HOP')[1].split('L')[0]
L = directory[2].split('L')[1].split('lr')[0]
decoder = "DAHIMN"
BSZ =  int(directory[2].split('BSZ')[1].split('DR')[0])

if 'kvr' in directory[1].split('-')[1].lower():
	DS = 'kvr' 
elif 'cam' in directory[1].split('-')[1].lower():
	DS = 'cam'
else:
	print('[ERROR] dataset name is None')
	exit()

print("HOP: ", HOP)
print("Layer: ", L)

if DS=='kvr': 
    from utils.utils_Ent_kvr import *
elif DS=='babi':
    from utils.utils_Ent_babi import *
elif DS == 'cam':
	from utils.utils_Ent_cam import *
else: 
    print("You need to provide the --dataset information")

train, dev, test, testOOV, lang, max_resp_len = prepare_data_seq(task, batch_size=BSZ)

model = globals()[decoder](
	int(HDD), 
	lang, 
	max_resp_len, 
	args['path'], 
	"", 
	DS,
	lr=0.0, 
	hop=int(HOP), 
	n_layer=int(L),
	dropout=0.0)

acc_test = model.evaluate(test, 1e7) 
if testOOV!=[]: 
	acc_oov_test = model.evaluate(testOOV, 1e7)