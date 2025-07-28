# 在程序的顶部或初始化部分

from datasets import load_from_disk
import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch import nn
from tqdm import tqdm
from tokenizer import CustomT5Tokenizer
from model_t5_loadt5bert_addencoder import T5Gan, T5Discriminator
from utils import evaluate_with_generate_bert
from model_confs import model_confs
import random
import numpy as np
import torch
import time
import model_glo as glo
from utils import data_augmentation
from transformers import T5EncoderModel 
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
parser.add_argument('--tokenizer_path', type=str, default='', help='tokenizer path')

parser.add_argument('--model_path', type=str, default=None, help='encoder_path')
parser.add_argument('--t5_encoder_path', type=str, default=None, help='mlm_layer_path')
parser.add_argument('--t5_gan_model_path', type=str, default=None, help='t5_gan_model_path')

parser.add_argument('--train_d', type=str, default='../our_model/dataset/t5_train_len50_keep25', help='t5_train_len50_keep25 t5_train_len65_keep25')
parser.add_argument('--val_d', type=str, default='../our_model/dataset/t5_val_len50_keep25', help='t5_val_len50_keep25 t5_val_len65_keep25')
parser.add_argument('--test_d', type=str, default='../our_model/dataset/t5_test_len50_keep25', help='t5_test_len50_keep25 t5_test_len65_keep25')
parser.add_argument('--use_generate',  action="store_true", help='use_generate')
parser.add_argument('--use_discriminator',  action="store_true", help='use_discriminator')
parser.add_argument('--use_pretrain_model',  action="store_true", help='use_pretrain_model')
parser.add_argument('--use_generate_output',  action="store_true", help='use_pretrain_model')
parser.add_argument('--use_generate_num', type=int, default=20, help='use_generate_num num')
parser.add_argument('--epoch_num', type=int, default=60, help='epoch num')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--max_length', type=int, default=42, help='max_length')
parser.add_argument('--gen_lr', type=float, default=1e-4, help='gen_lr')
parser.add_argument('--dis_lr', type=float, default=1e-4, help='gen_lr')
parser.add_argument('--tolerance', type=float, default=0.05, help='tolerance')

parser.add_argument('--early_stop', type=int, default=1000, help='early_stop')
parser.add_argument('--g_r', type=float, default=1.0, help='generate loss rate')
parser.add_argument('--d_r', type=float, default=1.0, help='discrimitor loss rate')

parser.add_argument('--data_per', type=float, default=1.0, help='discrimitor loss rate')

parser.add_argument('--save_bert_path', type=str, default=None, help='mlm_layer_path')


# model config
parser.add_argument('--hidden_size', type=int, default=256, help='model config')
parser.add_argument('--d_kv', type=int, default=64, help='model config')
parser.add_argument('--num_layers', type=int, default=4, help='model config')
parser.add_argument('--num_heads', type=int, default=4, help='model config')
parser.add_argument('--relative_attention_num_buckets', type=int, default=32, help='model config')
parser.add_argument('--num_decoder_layers', type=int, default=4, help='model config')
parser.add_argument('--d_ff', type=int, default=1024, help='model config')


opts = parser.parse_args()
print(vars(opts))
#
model_conf = model_confs( opts.hidden_size, opts.d_kv, opts.d_ff, opts.num_layers, opts.num_heads, opts.relative_attention_num_buckets, opts.num_decoder_layers)

# 载入自定义的tokenizer
tokenizer_path = opts.tokenizer_path#
train_d = load_from_disk(opts.train_d)#
val_d = load_from_disk(opts.val_d)#
test_d = load_from_disk(opts.test_d)#
model_path =  opts.model_path#
t5_encoder_path =  opts.t5_encoder_path#
t5_gan_model_path =  opts.t5_gan_model_path#
tokenizer = CustomT5Tokenizer.from_pretrained(tokenizer_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch_num = opts.epoch_num
batch_size = opts.batch_size
max_length = opts.max_length
early_stop = opts.early_stop

use_generate = opts.use_generate
use_discriminator = opts.use_discriminator
use_pretrain_model = opts.use_pretrain_model
use_generate_num = opts.use_generate_num
use_generate_output = opts.use_generate_output

g_r = opts.g_r 
d_r = opts.d_r

print(use_generate,use_generate_num)




def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # 使用 Xavier 正态分布初始化
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        # 对于 LayerNorm 层，一般初始化权重为1，偏差为0
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

discriminator = T5Discriminator(t5_encoder_path, tokenizer, device, model_conf)
if use_pretrain_model and os.path.exists(t5_gan_model_path):
    print('load model from', t5_gan_model_path)
    model = T5Gan.from_pretrained(t5_gan_model_path, tokenizer, device,model_conf,max_length=max_length,use_generate_output= use_generate_output,g_r =g_r ,d_r = d_r)
else:

    print('init model encoder decoder with pretrain encdoer weight')
    model = T5Gan(model_path, tokenizer, device, model_conf, max_length=max_length,use_generate_output=use_generate_output,g_r =g_r ,d_r = d_r)



bert = T5EncoderModel.from_pretrained(t5_encoder_path).to(device)

def collate_fn(batch):
    # Initialize the output batch with empty lists
    batch_output = {
        'input_ids': [],
        'attention_mask': [],
        'decoder_input_ids': [],
        'labels': []
    }

    # Iterate through each item and append the data to the respective lists
    for item in batch:
        batch_output['input_ids'].append(torch.tensor(item['input_ids']))
        batch_output['attention_mask'].append(torch.tensor(item['attention_mask']))
        
        # Prepare decoder_input_ids by shifting the labels to the right
        decoder_input_ids = [tokenizer.pad_token_id] + item['labels'][:-1]
        batch_output['decoder_input_ids'].append(torch.tensor(decoder_input_ids))

        # Prepare labels, replacing pad token id with -100 for ignoring in loss computation
        labels = torch.tensor(item['labels'])
        labels[labels == tokenizer.pad_token_id] = -100
        batch_output['labels'].append(labels)

    # Stack the lists to create tensors
    batch_output['input_ids'] = torch.stack(batch_output['input_ids'])
    batch_output['attention_mask'] = torch.stack(batch_output['attention_mask'])
    batch_output['decoder_input_ids'] = torch.stack(batch_output['decoder_input_ids'])
    batch_output['labels'] = torch.stack(batch_output['labels'])

    return batch_output

## 缩减训练数据
if opts.data_per < 1.0:
    train_size = int(opts.data_per * len(train_d))
    val_size = len(train_d) - train_size
    # 使用 random_split 分割数据集
    train_d, _ = random_split(train_d, [train_size, val_size])

print('train size:',len(train_d))


train_d = DataLoader(train_d, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_d = DataLoader(val_d, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_d = DataLoader(test_d, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


#train
model.to(device)
discriminator.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=opts.gen_lr) #只对生成器权重做优化

dis_optimizer = AdamW(discriminator.parameters(), lr=opts.dis_lr) #只对判别器权重做优化
bert_optimizer = AdamW(bert.parameters(), lr=opts.gen_lr) #只对判别器权重做优化
#使用余弦退火
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=0)
dis_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dis_optimizer, T_max=500, eta_min=0)


st = time.time()
early_num = 0
early_recall = 0


best_recall = 0
b_acc = 0
b_pre = 0
b_epoch = 0
    
# 阶段标记
generator_stable = False
discriminator_stable = False  if opts.use_discriminator else True



# 滑动窗口和阈值
window_size = 3
tolerance = opts.tolerance

# 判断变化率
def is_stable(losses, window_size, tolerance=1e-3):
    if len(losses) < 2 * window_size:
        return False
    window_1 = losses[-2 * window_size:-window_size]
    window_2 = losses[-window_size:]
    mean_1 = sum(window_1) / window_size
    mean_2 = sum(window_2) / window_size
   
    change_rate = abs(mean_1 - mean_2) / mean_1  # 如果loss 增大 也视为稳定
    print('change_rate', change_rate)
    return change_rate < tolerance

# 训练循环
gen_losses = []
dis_losses = []

# train_ = False
train_ = True




glo._init()

model.eval()
avg_acc, avg_recall, avg_precision = evaluate_with_generate_bert(model.generator.model, val_d, tokenizer, device, max_length, True,bert)
print('zero shot')
print(f'Average Accuracy: {avg_acc}')
print(f'Average Recall: {avg_recall}')
print(f'Average Precision: {avg_precision}')
model.train()

for epoch in range(epoch_num):
    ## Train generator without discriminator feedback
    if (not generator_stable  or not  opts.use_discriminator)and train_:
        dis_losses = []  # 清空判别器 loss list
        print('train generator wo dis loss')
        for param in model.generator.parameters():
            param.requires_grad = True
        for param in discriminator.parameters():
            param.requires_grad = False

        model.zero_grad()
        model.train()
        bert.zero_grad()
        bert.train()
        # bert.eval()
        progress_bar = tqdm(train_d)
        avg_gen_loss = 0

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            global_bert_hiddenstate = bert(input_ids, attention_mask=attention_mask).last_hidden_state#.detach()
            glo.set_value('global_bert_hiddenstate',global_bert_hiddenstate)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)

            loss = outputs.loss
            avg_gen_loss += loss.item()
            optimizer.zero_grad()
            bert_optimizer.zero_grad()
            loss.backward()
            bert_optimizer.step() 
            optimizer.step()

            lr_scheduler.step()

            progress_bar.set_description(f'generator epoch:{epoch} loss: {loss.item()} lr:{lr_scheduler.get_last_lr()}')

        avg_gen_loss /= len(train_d)
        gen_losses.append(avg_gen_loss)
        print(f'generator epoch:{epoch} avg loss: {round(avg_gen_loss, 5)}')

        # Evaluate generator stability
        if is_stable(gen_losses, window_size, tolerance):
            generator_stable = True

    ## Train discriminator until stable
    elif not discriminator_stable and train_ and  opts.use_discriminator:
        gen_losses = []  # 清空生成器 loss list

        print('train discriminator')
     
        for param in discriminator.parameters():
            param.requires_grad = True

        discriminator.zero_grad()
        discriminator.train()
        progress_bar = tqdm(train_d)
        avg_dis_loss = 0

        for i,batch in enumerate(progress_bar):
            sub_batch_size = len(batch['input_ids'])

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)

            r_inputs = labels.clone()
            discriminator_attention  = (r_inputs != -100).int()  # 创建 discriminator_attention
            pad_token_id = tokenizer.pad_token_id
            r_inputs[r_inputs == -100] = pad_token_id #用于训练判别器
            
            
            with torch.no_grad():
                global_bert_hiddenstate = bert(input_ids, attention_mask=attention_mask).last_hidden_state.detach()
                glo.set_value('global_bert_hiddenstate',global_bert_hiddenstate)
                # model.eval()
                if use_generate_output:
                    gen_outputs = model.get_generate_output(input_ids=input_ids, attention_mask=attention_mask)
                    fake_inputs = gen_outputs
                else:
                    gen_outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
                    fake_inputs = torch.argmax(gen_outputs.logits, dim=-1)

                r_inputs = model.generator.model.shared(r_inputs).detach()
                fake_inputs = model.generator.model.shared(fake_inputs).detach()

                # model.train()

            r_labels = torch.ones(sub_batch_size, dtype=torch.long).to(device)
            r_outputs =discriminator(embedded_inputs=r_inputs, attention_mask=discriminator_attention, labels=r_labels)

            fake_labels = torch.zeros(sub_batch_size, dtype=torch.long).to(device)
            f_outputs =discriminator(embedded_inputs=fake_inputs, attention_mask=discriminator_attention, labels=fake_labels)

            dis_loss = (r_outputs.loss + f_outputs.loss) / 2
            avg_dis_loss += dis_loss.item()
            dis_optimizer.zero_grad()
            dis_loss.backward()
            dis_optimizer.step()
            dis_lr_scheduler.step()
            progress_bar.set_description(f'discriminator epoch:{epoch} loss: {dis_loss.item()} lr:{dis_lr_scheduler.get_last_lr()}')

        avg_dis_loss /= len(train_d)
        dis_losses.append(avg_dis_loss)
        print(f'discriminator epoch:{epoch} avg loss: {round(avg_dis_loss, 5)}')

        # Evaluate discriminator stability
        if is_stable(dis_losses, window_size, tolerance):
            discriminator_stable = True

    ## Train generator with discriminator feedback
    elif train_ and  opts.use_discriminator:
        dis_losses = []  # 清空判别器 loss list
        print('train generator with dis loss')
        for param in model.generator.parameters():
            param.requires_grad = True
        for param in discriminator.parameters():
            param.requires_grad = False

        model.zero_grad()
        bert.zero_grad()
        bert.train()
        model.train()
        progress_bar = tqdm(train_d)
        avg_gen_loss = 0
        avg_dis_loss = 0

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)

            global_bert_hiddenstate = bert(input_ids, attention_mask=attention_mask).last_hidden_state
            glo.set_value('global_bert_hiddenstate',global_bert_hiddenstate)

            discriminator_attention  = (labels != -100).int()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels, use_discriminator=True, discriminator_attention=discriminator_attention,discriminator=discriminator)

            loss = outputs.loss
            avg_gen_loss += loss.item()
            if hasattr(outputs, 'dis_loss'):
                avg_dis_loss += outputs.dis_loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            progress_bar.set_description(f'generator epoch:{epoch} loss: {loss.item()} dis_loss: {outputs.dis_loss.item() if hasattr(outputs, "dis_loss") else "N/A"} lr:{lr_scheduler.get_last_lr()}')

        avg_gen_loss /= len(train_d)
        avg_dis_loss /= len(train_d)
        gen_losses.append(avg_gen_loss)
        
        print(f'generator epoch:{epoch} avg loss: {round(avg_gen_loss, 5)} dis_loss: {round(avg_dis_loss, 8)}')

        # Evaluate discriminator stability
        # if is_stable(dis_losses, window_size, tolerance):
        if is_stable(gen_losses, window_size, tolerance):
            discriminator_stable = False

    # 评估
    if discriminator_stable:  # 不训练判别器时再进行评估
        model.eval()
        eva_t = time.time()
        avg_acc, avg_recall, avg_precision = evaluate_with_generate_bert(model.generator.model, val_d, tokenizer, device, max_length, True,bert)
        print('eval time', round(time.time() - eva_t, 4))
        print('epoch: ', epoch)
        print(f'Average Accuracy: {avg_acc}')
        print(f'Average Recall: {avg_recall}')
        print(f'Average Precision: {avg_precision}')

        if best_recall <= avg_recall:
            best_recall = avg_recall
            b_acc = avg_acc
            b_pre = avg_precision
            b_epoch = epoch
            model.save_pretrained(t5_gan_model_path)
            if  opts.save_bert_path is not None:
                bert.save_pretrained( opts.save_bert_path)
            

        model.train()

        # early stop
        if early_recall > avg_recall:
            early_num += 1
        else:
            early_recall = avg_recall
            early_num = 0
        if early_num > early_stop:
            break

print('took ',round(time.time()-st,2),' s')
# model.save_pretrained(t5_gan_model_path)
print('====test result====')
model.eval()
avg_acc,avg_recall,avg_precision = evaluate_with_generate_bert(model.generator.model, test_d, tokenizer, device, max_length,True,bert)
print(f'test Average Accuracy: {avg_acc}')
print(f'test Average Recall: {avg_recall}')
print(f'test Average Precision: {avg_precision}')

print('----best result----')
print('b_epoch',b_epoch)
print('b_acc',b_acc)
print('b_precision',b_pre)
print('b_recall',best_recall)

