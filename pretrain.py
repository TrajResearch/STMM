from model import T5MLM
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import random
import numpy as np
from tokenizer import CustomT5Tokenizer
from torch.utils.data import DataLoader, Dataset
import os
import time
from datasets import load_dataset,load_from_disk
from model_confs import model_confs
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import argparse
parser = argparse.ArgumentParser(description='our t5 pre train')
parser.add_argument('--tokenizer_path', type=str, default='../our_model/merge_xrx_tokenizer', help='tokenizer path')
parser.add_argument('--modelpath', type=str, default='./t5_len50_keep25', help='model path')

parser.add_argument('--encoder_path', type=str, default='model/encoder_path', help='encoder_path')
parser.add_argument('--mlm_layer_path', type=str, default='model/mlm_layer_path', help='mlm_layer_path')
parser.add_argument('--t5_encoder_path', type=str, default='model/t5_encoder_path', help='t5_encoder_path')

parser.add_argument('--train_d', type=str, default='', help='train path')
parser.add_argument('--val_d', type=str, default='', help='train path')

parser.add_argument('--ifLoadpretrain', type=bool, default=False, help='ifLoadpretrain')
parser.add_argument('--epoch_num', type=int, default=20, help='epoch num')
parser.add_argument('--batch_size', type=int, default=128, help='epoch num')

# model config
parser.add_argument('--hidden_size', type=int, default=256, help='model config')
parser.add_argument('--d_kv', type=int, default=64, help='model config')
parser.add_argument('--num_layers', type=int, default=4, help='model config')
parser.add_argument('--num_heads', type=int, default=4, help='model config')
parser.add_argument('--relative_attention_num_buckets', type=int, default=32, help='model config')
parser.add_argument('--num_decoder_layers', type=int, default=4, help='model config')
parser.add_argument('--d_ff', type=int, default=1024, help='model config')

opts = parser.parse_args()


model_conf = model_confs( opts.hidden_size, opts.d_kv, opts.d_ff, opts.num_layers, opts.num_heads, opts.relative_attention_num_buckets, opts.num_decoder_layers)

# 载入自定义的tokenizer

tokenizer_path = opts.tokenizer_path
tokenizer = CustomT5Tokenizer.from_pretrained(tokenizer_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
ifLoadpretrain = opts.ifLoadpretrain
model_path = None
epoch_num = opts.epoch_num
batch_size = opts.batch_size

train_dataset = load_from_disk(opts.train_d)
val_dataset = load_from_disk(opts.val_d)

encoder_path = opts.encoder_path
mlm_layer_path = opts.mlm_layer_path
t5_encoder_path = opts.t5_encoder_path
model = T5MLM(model_path, tokenizer, device, model_conf)



def collate_fn(batch):
    # Initialize the output batch with empty lists
    batch_output = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }

    # Iterate through each item and append the data to the respective lists
    for item in batch:
        batch_output['input_ids'].append(torch.tensor(item['input_ids']))
        batch_output['attention_mask'].append(torch.tensor(item['attention_mask']))
        
        # Prepare decoder_input_ids by shifting the labels to the right
        decoder_input_ids = [tokenizer.pad_token_id] + item['labels'][:-1]

        # Prepare labels, replacing pad token id with -100 for ignoring in loss computation
        labels = torch.tensor(item['labels'])
        # labels[labels == tokenizer.pad_token_id] = -100
        batch_output['labels'].append(labels)

    # Stack the lists to create tensors
    batch_output['input_ids'] = torch.stack(batch_output['input_ids'])
    batch_output['attention_mask'] = torch.stack(batch_output['attention_mask'])
    batch_output['labels'] = torch.stack(batch_output['labels'])

    return batch_output


st = time.time()
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
optimizer = AdamW(model.parameters(), lr=1e-4)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
#使用余弦退火
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0)

model.train()


def test(model,data_loader):
    model.eval()
    total_loss = 0
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], labels=batch['labels'],attention_mask=batch['attention_mask'])
        total_loss += outputs.loss.item()
    

    model.train()
    return total_loss/len(data_loader)


best_test_loss = 100
early_cnt = 0
for epoch in range(epoch_num):  # 
    progress_bar = tqdm(range(len(train_dataloader)))
    for i, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        if epoch == 0 and i == 0:
            print('input',batch['input_ids'][0])
            print('labels',batch['labels'][0])
            print('attention_mask',batch['attention_mask'][0])

        outputs = model(input_ids=batch['input_ids'], labels=batch['labels'],attention_mask=batch['attention_mask'])
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        progress_bar.update(1)
        progress_bar.set_description(f'epoch {epoch} loss: {loss.item()} lr:{lr_scheduler.get_last_lr()}')
    
    test_loss = test(model,val_dataloader)
    print('epoch',epoch,'test loss',test_loss)
    if best_test_loss >= test_loss:
        best_test_loss = test_loss
        early_cnt = 0
        model.save_model(mlm_layer_path = mlm_layer_path, t5_encoder_path = t5_encoder_path,encoder_path = encoder_path)
    else:
        early_cnt += 1
        if early_cnt > 10:break  #早停    


print('train cost:', time.time()-st)
# model.save_model(mlm_layer_path = mlm_layer_path, t5_encoder_path = t5_encoder_path,encoder_path = encoder_path)



