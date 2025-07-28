import torch
from loss import cal_pre_recall, cal_pre_recall_2,cal_pre_recall_3
import model_glo as glo
from torch_geometric.data import Data

from torch_geometric.utils import  to_undirected

glo._init()
# 定义损失函数
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

def get_gat_hiddenstate( input_ids):
    gat_z = glo.get_value('gat_z')
    if gat_z is None:
        gat_z = torch.load('gat_hidden_states.pt')
        glo.set_value('gat_z',gat_z)
    assert gat_z is not None, 'gat_z is None, please set it before forward' 

    input_ids = input_ids.to(gat_z.device)
    hidden_state_sequences = []
    for node_index_sequence in input_ids:
        hidden_state_sequence = gat_z[node_index_sequence.to(gat_z.device)]
        hidden_state_sequences.append(hidden_state_sequence)
    hidden_state_sequences = torch.stack(hidden_state_sequences)

    return hidden_state_sequences

def cal_loss(outputs,target_ids):
    shift_logits = outputs[:, 1:].contiguous().float()
    shift_labels = target_ids[:,:-1].contiguous().float()

    # 计算损失
    loss = loss_fct(shift_logits.view(-1), shift_labels.view(-1))
    print("generate Loss:", loss.item())


def evaluate_with_generate( model, test_dataloader, tokenizer, device, max_length,if_print=False):
    total_acc, total_recall, total_precision = 0, 0, 0
    total_count = 0
    with torch.no_grad():
        model.eval()
        for i,batch in enumerate(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            true_labels = batch['labels'].to(device)

            # Generate text using the model
          
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length,decoder_start_token_id=tokenizer.pad_token_id,top_k=1,
                do_sample=True,
                temperature=0.01,)

            # Ensure the generated ids and true labels have the same shape
            if generated_ids.shape[1] < true_labels.shape[1]:
                # Padding generated_ids to match true_labels length
                padding = torch.full((generated_ids.shape[0], true_labels.shape[1] - generated_ids.shape[1]), tokenizer.pad_token_id, device=device)
                generated_ids = torch.cat([generated_ids, padding], dim=1)
            else:
                # Truncating generated_ids to match true_labels length
                generated_ids = generated_ids[:, :true_labels.shape[1]]
            generated_ids = generated_ids[:, 1:] # generate 跳过第一个token
            if if_print and i == 0:
                print('input ids',input_ids[0])
                print('generated_ids ',generated_ids[0])
                print('true labels ',true_labels[0])
            acc, recall, precision = cal_pre_recall(generated_ids, true_labels)
            total_acc += acc
            total_recall += recall
            total_precision += precision
            total_count += 1
            
        
        avg_acc = total_acc / total_count
        avg_recall = total_recall / total_count
        avg_precision = total_precision / total_count
        model.train()

    return avg_acc,avg_recall,avg_precision

def  evaluate_with_generate_gat_train( model, test_dataloader, tokenizer, device, max_length,if_print=False,gat=None,indices_tensor = None,edge_index=None):

    total_acc, total_recall, total_precision = 0, 0, 0

    indices_tensor = indices_tensor
    edge_index=edge_index

    total_count = 0
    model.eval()
    gat.eval()

    with torch.no_grad():
        
        for i,batch in enumerate(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            true_labels = batch['labels'].to(device)

            ### gat 
            gat_x = model.shared(indices_tensor) # 获取 embedding
            #根据索引找到 rid 对应的 embedding
            #构造 gat data
            gatdata = Data(x=gat_x, edge_index=edge_index)
            gatdata.edge_index = to_undirected(gatdata.edge_index) # 无向图
            gat_embedding = gat(gatdata.x, gatdata.edge_index) # shape [num_nodes, 256] extend to  batch size, num_nodes,256
            #赋值到全局变量
            glo.set_value('global_gat_hiddenstate_new',gat_embedding)


            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length,decoder_start_token_id=tokenizer.pad_token_id,top_k=1,
                do_sample=True,
                temperature=0.01,)
       

            # Ensure the generated ids and true labels have the same shape
            if generated_ids.shape[1] < true_labels.shape[1]:
                # Padding generated_ids to match true_labels length
                padding = torch.full((generated_ids.shape[0], true_labels.shape[1] - generated_ids.shape[1]), tokenizer.pad_token_id, device=device)
                generated_ids = torch.cat([generated_ids, padding], dim=1)
            else:
                # Truncating generated_ids to match true_labels length
                generated_ids = generated_ids[:, :true_labels.shape[1]]
            generated_ids = generated_ids[:, 1:] # generate 跳过第一个token
            if if_print and i == 0:
                print('input ids',input_ids[0])
                print('generated_ids ',generated_ids[0])
                print('true labels ',true_labels[0])
            acc, recall, precision = cal_pre_recall(generated_ids, true_labels)
            total_acc += acc
            total_recall += recall
            total_precision += precision
            total_count += 1
           
        
        avg_acc = total_acc / total_count
        avg_recall = total_recall / total_count
        avg_precision = total_precision / total_count
        model.train()
        gat.train()
    return avg_acc,avg_recall,avg_precision

def evaluate_with_generate_gat( model, test_dataloader, tokenizer, device, max_length,if_print=False):
    total_acc, total_recall, total_precision = 0, 0, 0
    total_count = 0
    model.eval()

    with torch.no_grad():
        
        for i,batch in enumerate(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            true_labels = batch['labels'].to(device)

         
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length,decoder_start_token_id=tokenizer.pad_token_id,top_k=1,
                do_sample=True,
                temperature=0.01,)
     

            # Ensure the generated ids and true labels have the same shape
            if generated_ids.shape[1] < true_labels.shape[1]:
                # Padding generated_ids to match true_labels length
                padding = torch.full((generated_ids.shape[0], true_labels.shape[1] - generated_ids.shape[1]), tokenizer.pad_token_id, device=device)
                generated_ids = torch.cat([generated_ids, padding], dim=1)
            else:
                # Truncating generated_ids to match true_labels length
                generated_ids = generated_ids[:, :true_labels.shape[1]]
            generated_ids = generated_ids[:, 1:] # generate 跳过第一个token
            if if_print and i == 0:
                print('input ids',input_ids[0])

                print('generated_ids ',generated_ids[0])
                print('true labels ',true_labels[0])
            acc, recall, precision = cal_pre_recall(generated_ids, true_labels)
            total_acc += acc
            total_recall += recall
            total_precision += precision
            total_count += 1
       
        
        avg_acc = total_acc / total_count
        avg_recall = total_recall / total_count
        avg_precision = total_precision / total_count
        model.train()
        

    return avg_acc,avg_recall,avg_precision

def evaluate_with_generate_bert( model, test_dataloader, tokenizer, device, max_length,if_print=False,bert=None):
    total_acc, total_recall, total_precision = 0, 0, 0
    total_count = 0
    with torch.no_grad():
        model.eval()
        bert.eval()
        for i,batch in enumerate(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            true_labels = batch['labels'].to(device)

            global_bert_hiddenstate = bert(input_ids, attention_mask=attention_mask).last_hidden_state.detach()
            
            glo.set_value('global_bert_hiddenstate',global_bert_hiddenstate)
            # Generate text using the model

            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length,decoder_start_token_id=tokenizer.pad_token_id,top_k=1,
                do_sample=True,
                temperature=0.01,)
            

            # Ensure the generated ids and true labels have the same shape
            if generated_ids.shape[1] < true_labels.shape[1]:
                # Padding generated_ids to match true_labels length
                padding = torch.full((generated_ids.shape[0], true_labels.shape[1] - generated_ids.shape[1]), tokenizer.pad_token_id, device=device)
                generated_ids = torch.cat([generated_ids, padding], dim=1)
            else:
                # Truncating generated_ids to match true_labels length
                generated_ids = generated_ids[:, :true_labels.shape[1]]
            generated_ids = generated_ids[:, 1:] # generate 跳过第一个token
            if if_print and i == 0:
                print('input ids',input_ids[0])

                print('generated_ids ',generated_ids[0])
                print('true labels ',true_labels[0])
            acc, recall, precision = cal_pre_recall(generated_ids, true_labels)
            total_acc += acc
            total_recall += recall
            total_precision += precision
            total_count += 1
            # print(acc)
        
        avg_acc = total_acc / total_count
        avg_recall = total_recall / total_count
        avg_precision = total_precision / total_count
        model.train()
        bert.train()

    return avg_acc,avg_recall,avg_precision

def evaluate_with_generate2( model, test_dataloader, tokenizer, device, max_length,if_print=False):
    total_acc, total_recall, total_precision = 0, 0, 0
    total_recall2, total_precision2 = 0, 0
    total_count = 0
    with torch.no_grad():
        model.eval()
        for i,batch in enumerate(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            true_labels = batch['labels'].to(device)

            # Generate text using the model
            
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length,decoder_start_token_id=tokenizer.pad_token_id,top_k=1,
                do_sample=True,
                temperature=0.01,)

            # Ensure the generated ids and true labels have the same shape
            if generated_ids.shape[1] < true_labels.shape[1]:
                # Padding generated_ids to match true_labels length
                padding = torch.full((generated_ids.shape[0], true_labels.shape[1] - generated_ids.shape[1]), tokenizer.pad_token_id, device=device)
                generated_ids = torch.cat([generated_ids, padding], dim=1)
            else:
                # Truncating generated_ids to match true_labels length
                generated_ids = generated_ids[:, :true_labels.shape[1]]
            generated_ids = generated_ids[:, 1:] # generate 跳过第一个token
            if if_print and i == 0:
                print('generated_ids ',generated_ids[0])
                print('true labels ',true_labels[0])
            acc, recall, precision = cal_pre_recall(generated_ids, true_labels)
            recall2, precision2 = cal_pre_recall_2(generated_ids, true_labels)
            total_acc += acc
            total_recall += recall
            total_precision += precision
            total_recall2 += recall2
            total_precision2 += precision2
            total_count += 1
            # print(acc)
        
        avg_acc = total_acc / total_count
        avg_recall = total_recall / total_count
        avg_precision = total_precision / total_count
        avg_recall2 = total_recall2 / total_count
        avg_precision2 = total_precision2 / total_count
        model.train()
    
    return avg_acc,avg_recall,avg_precision, avg_recall2, avg_precision2


def evaluate_with_generate3( model, test_dataloader, tokenizer, device, max_length,if_print=False):
    total_acc, total_recall, total_precision = 0, 0, 0
    total_recall2, total_precision2 = 0, 0
    total_recall3, total_precision3 = 0, 0
    total_lcs = 0
    total_count = 0
    with torch.no_grad():
        model.eval()
        for i,batch in enumerate(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            true_labels = batch['labels'].to(device)

            # Generate text using the model
            
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length,decoder_start_token_id=tokenizer.pad_token_id,top_k=1,
                do_sample=True,
                temperature=0.01,)

            # Ensure the generated ids and true labels have the same shape
            if generated_ids.shape[1] < true_labels.shape[1]:
                # Padding generated_ids to match true_labels length
                padding = torch.full((generated_ids.shape[0], true_labels.shape[1] - generated_ids.shape[1]), tokenizer.pad_token_id, device=device)
                generated_ids = torch.cat([generated_ids, padding], dim=1)
            else:
                # Truncating generated_ids to match true_labels length
                generated_ids = generated_ids[:, :true_labels.shape[1]]
            generated_ids = generated_ids[:, 1:] # generate 跳过第一个token
            if if_print and i == 0:
                print('generated_ids ',generated_ids[0])
                print('true labels ',true_labels[0])
            # acc, recall, precision = cal_pre_recall(generated_ids, true_labels)
            acc, recall, precision, precision2,recall2,precision3,recall3,lcs = cal_pre_recall_3(generated_ids, true_labels)
            
            total_acc += acc
            total_recall += recall
            total_precision += precision
            total_recall2 += recall2
            total_precision2 += precision2
            total_recall3 += recall3
            total_precision3 += precision3
            total_lcs += lcs
            total_count += 1
        
        
        avg_acc = total_acc / total_count
        avg_recall = total_recall / total_count
        avg_precision = total_precision / total_count
        avg_recall2 = total_recall2 / total_count
        avg_precision2 = total_precision2 / total_count
        avg_recall3 = total_recall3 / total_count
        avg_precision3 = total_precision3 / total_count
        avg_lcs = total_lcs / total_count

        model.train()
    
    return avg_acc,avg_recall,avg_precision, avg_recall2, avg_precision2, avg_recall3, avg_precision3, avg_lcs


def data_augmentation(input_ids, attention_mask, tokenizer ,drop_prob=0.1):
    #随机一个 drop prob
    
    drop_prob = 0.0 + torch.rand(1).item() * (0.15 - 0.00) #drop 1% 到 15%
    batch_size, seq_len = input_ids.size()
    pad_token_id = tokenizer.pad_token_id

    # 生成一个mask，用于标记应避免丢弃的特殊token位置
    special_tokens_mask = torch.zeros_like(input_ids).bool()
    for special_id in tokenizer.all_special_ids:
        special_tokens_mask = special_tokens_mask | (input_ids == special_id)
    special_tokens_mask = special_tokens_mask.to(input_ids.device)
    # 确定每个token是否被丢弃，同时确保特殊token不被丢弃
    drop_mask = (torch.rand(input_ids.shape).to(input_ids.device) < drop_prob) & ~special_tokens_mask

    # 创建一个索引，用于收集未被丢弃的token
    idx = torch.arange(seq_len).repeat(batch_size, 1).to(input_ids.device)
    idx_masked = idx * (1 - drop_mask.long()) + (drop_mask.long() * seq_len)
    idx_sorted = idx_masked.argsort()

    # 收集未被丢弃的token，并在末尾添加pad
    input_ids_dropped = torch.gather(input_ids, 1, idx_sorted)
    input_ids_dropped[:, -drop_mask.sum(dim=1).max():] = pad_token_id

    attention_mask_dropped = torch.gather(attention_mask, 1, idx_sorted)
    attention_mask_dropped[:, -drop_mask.sum(dim=1).max():] = 0

    return input_ids_dropped, attention_mask_dropped



def gps2grid(self, lat,lng, max_lat,min_lat,max_lng,min_lng, grid_size = 50):
        """
        'min_lat':36.6456,
        'min_lng':116.9854,
        'max_lat':36.6858,
        'max_lng':117.0692,
        
        grid size:
            int. in meter
        """
        LAT_PER_METER = 8.993203677616966e-06
        LNG_PER_METER = 1.1700193970443768e-05
        lat_unit = LAT_PER_METER * grid_size
        lng_unit = LNG_PER_METER * grid_size
        
        max_xid = int((max_lat - min_lat) / lat_unit) + 1
        max_yid = int((max_lng - min_lng) / lng_unit) + 1
        
        
        locgrid_x = int((lat - min_lat) / lat_unit) + 1
        locgrid_y = int((lng - min_lng) / lng_unit) + 1
        
        return locgrid_x, locgrid_y