from tqdm import tqdm

from utils.config import *
from models.DAHIMN import *
import torch
import numpy as np 
import random




early_stop = args['earlyStop']
if args['dataset']=='kvr':
    from utils.utils_Ent_kvr import *
    # early_stop = 'BLEU'
elif args['dataset'] == 'cam':
    from utils.utils_Ent_cam import *
elif args['dataset']=='babi':
    from utils.utils_Ent_babi import *
    early_stop = None 
    if args["task"] not in ['1','2','3','4','5']:
        print("[ERROR] You need to provide the correct --task information")
        exit(1)
else:
    print("[ERROR] You need to provide the --dataset information")

avg_best, cnt, acc = 0.0, 0, 0.0
train, dev, test, testOOV, lang, max_resp_len = prepare_data_seq(args['task'], batch_size=int(args['batch']))

model = globals()[args['decoder']](
    int(args['hidden']), 
    lang, 
    max_resp_len, 
    args['path'], 
    args['task'],
    args['dataset'],
    lr=float(args['learn']),
    hop=int(args['hop']),
    n_layer=int(args['n_layer']), 
    dropout=float(args['drop']))

for epoch in range(200):
    print("Epoch:{}".format(epoch))  
    # Run the train function
    pbar = tqdm(enumerate(train),total=len(train))
    for i, data in pbar:
        model.train_batch(data, int(args['clip']), reset=(i==0))
        pbar.set_description(model.print_loss())
        # break
    if((epoch+1) % int(args['evalp']) == 0):    
        acc = model.evaluate(dev, avg_best, early_stop)
        model.scheduler.step(acc)

        if(acc >= avg_best):
            avg_best = acc
            cnt = 0
        else:
            cnt += 1

        if(cnt == 25 or (acc==1.0 and early_stop==None)): 
            print("Ran out of patient, early stop...")  
            break 


