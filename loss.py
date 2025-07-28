import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
sys.setrecursionlimit(500000)
#计算函数
def memoize(fn):
    '''Return a memoized version of the input function.

    The returned function caches the results of previous calls.
    Useful if a function call is expensive, and the function 
    is called repeatedly with the same arguments.
    '''
    cache = dict()
    def wrapped(*v):
        key = tuple(v) # tuples are hashable, and can be used as dict keys
        if key not in cache:
            cache[key] = fn(*v)
        return cache[key]
    return wrapped

def lcs(xs, ys):
    '''Return the longest subsequence common to xs and ys.

    Example
    >>> lcs("HUMAN", "CHIMPANZEE")
    ['H', 'M', 'A', 'N']
    '''
    @memoize
    def lcs_(i, j):
        if i and j:
            xe, ye = xs[i-1], ys[j-1]
            if xe == ye:
                return lcs_(i-1, j-1) + [xe]
            else:
                return max(lcs_(i, j-1), lcs_(i-1, j), key=len)
        else:
            return []
    return lcs_(len(xs), len(ys))

def shrink_seq(seq):
    '''
    去掉重复的 id
    '''
    s0 = seq[0]
    new_seq = [s0]
    for s in seq[1:]:
        if s == s0:
            continue
        else:
            new_seq.append(s)
        s0 = s
    
    return new_seq


def cal_pre_recall(batch_pre,batch_label,eos_token_id = 1):
    '''
    输入batch 数据，
    遍历 hmmm 输出文件，解析出每个文件 road id 序列： hmmm_list
    找到对应的文件，解析出 truth id 序列： truth_list
    计算对应位置的 相等的个数 ：cnt，
    统计全部id 个数 ：ttl
    truth_list,road_list  去重后的长度 ：ttl_trg_id_num ， ttl_pre_id_num
    统计truth_list,road_list  去重后的 shr_trg_ids,shr_pre_ids 最大公共长度：correct_id_num
    
    '''
    batch_pre = batch_pre.cpu().numpy().tolist()
    batch_label = batch_label.cpu().numpy().tolist()
    batch_size = len(batch_pre)

    cnt = 0
    ttl = 0
    ttl_trg_id_num = 0
    ttl_pre_id_num = 0
    correct_id_num = 0

    for num in range(batch_size):
        hmmm_list = batch_pre[num]
        truth_list = batch_label[num]

        #第一个 eos 位置
        for i in range(len(hmmm_list)):
            if truth_list[i] == eos_token_id  :# 
                hmmm_list = hmmm_list[:i]
                truth_list = truth_list[:i]
                break
            ttl+= 1
            if hmmm_list[i] == truth_list[i] :
                cnt += 1

        shr_trg_ids = shrink_seq(truth_list)
        shr_pre_ids = shrink_seq(hmmm_list)
        correct_id_num += len(lcs(shr_trg_ids, shr_pre_ids))
        ttl_trg_id_num += len(shr_trg_ids)
        ttl_pre_id_num += len(shr_pre_ids)
    
    rid_acc = cnt / ttl
    rid_recall = correct_id_num / ttl_trg_id_num
    rid_precision = correct_id_num / ttl_pre_id_num

    return rid_acc,rid_recall,rid_precision


def cal_pre_recall_2(batch_pre,batch_label,eos_token_id = 1):
    batch_pre = batch_pre.cpu().numpy().tolist()
    batch_label = batch_label.cpu().numpy().tolist()
    batch_size = len(batch_pre)

    p = 0
    r = 0
    for num in range(batch_size):
        y_pred = batch_pre[num]
        y = batch_label[num]
        #第一个 eos 位置
        y = y[:y.index(eos_token_id)]
        if eos_token_id in y_pred:
            y_pred = y_pred[:y_pred.index(eos_token_id)]
        else:
            y_pred = y_pred  
        intersect =set(y_pred)& set(y)
        p=p+ len(intersect)/ max(len(y_pred),len(y))
        if len(y_pred) == 0:
            r = r
        else:
            r=r+ len(intersect)/len(y_pred)

    return p/batch_size,r/batch_size

def cal_pre_recall_3(batch_pre,batch_label,eos_token_id = 1):
    batch_pre = batch_pre.cpu().numpy().tolist()
    batch_label = batch_label.cpu().numpy().tolist()
    batch_size = len(batch_pre)

    cnt = 0
    ttl = 0
    ttl_trg_id_num = 0
    ttl_pre_id_num = 0
    correct_id_num = 0

    p = 0
    r = 0
    p2,r2 = 0,0
    lcs1 = 0
    for num in range(batch_size):
        hmmm_list = batch_pre[num]
        truth_list = batch_label[num]

        #第一个 eos 位置
        for i in range(len(hmmm_list)):
            if truth_list[i] == eos_token_id  :# 
                hmmm_list = hmmm_list[:i]
                truth_list = truth_list[:i]
                break
            ttl+= 1
            if hmmm_list[i] == truth_list[i] :
                cnt += 1

        shr_trg_ids = shrink_seq(truth_list)
        shr_pre_ids = shrink_seq(hmmm_list)
        correct_id_num += len(lcs(shr_trg_ids, shr_pre_ids))
        ttl_trg_id_num += len(shr_trg_ids)
        ttl_pre_id_num += len(shr_pre_ids)

        #
        y_pred = batch_pre[num]
        y = batch_label[num]
        #第一个 eos 位置
        y = y[:y.index(eos_token_id)]
        if eos_token_id in y_pred:
            y_pred = y_pred[:y_pred.index(eos_token_id)]
        else:
            y_pred = y_pred  
        intersect =set(y_pred)& set(y)
        p= p + len(intersect)/ max(len(y_pred),len(y))

        p2=p2+ len(intersect)/ (len(set(y_pred)) if len(y_pred) != 0 else 1)
        if len(y_pred) == 0:
            r = r
        else:
            r = r+ len(intersect)/len(y_pred)
        r2 = r2+ len(intersect)/len(set(y))
    
    rid_acc = cnt / ttl
    rid_recall = correct_id_num / ttl_trg_id_num
    rid_precision = correct_id_num / ttl_pre_id_num
    lcs1 = correct_id_num / ttl

    return rid_acc, rid_recall, rid_precision, p/batch_size, r/batch_size, p2/batch_size, r2/batch_size, lcs1


def cal_pre_recall_list(batch_pre,batch_label):
    '''
    输入batch 数据，
    遍历 hmmm 输出文件，解析出每个文件 road id 序列： hmmm_list
    找到对应的文件，解析出 truth id 序列： truth_list
    计算对应位置的 相等的个数 ：cnt，
    统计全部id 个数 ：ttl
    truth_list,road_list  去重后的长度 ：ttl_trg_id_num ， ttl_pre_id_num
    统计truth_list,road_list  去重后的 shr_trg_ids,shr_pre_ids 最大公共长度：correct_id_num
    
    '''

    batch_size = len(batch_pre)

    cnt = 0
    ttl = 0
    ttl_trg_id_num = 0
    ttl_pre_id_num = 0
    correct_id_num = 0

    for num in range(batch_size):
        hmmm_list = batch_pre[num]
        truth_list = batch_label[num]

        #第一个 eos 位置
        for i in range(len(hmmm_list)):
            if i >= len(truth_list):break
          
            ttl+= 1
            if hmmm_list[i] == truth_list[i] :
                cnt += 1

        shr_trg_ids = shrink_seq(truth_list)
        shr_pre_ids = shrink_seq(hmmm_list)
        correct_id_num += len(lcs(shr_trg_ids, shr_pre_ids))
        ttl_trg_id_num += len(shr_trg_ids)
        ttl_pre_id_num += len(shr_pre_ids)
    
    rid_acc = cnt / ttl
    rid_recall = correct_id_num / ttl_trg_id_num
    rid_precision = correct_id_num / ttl_pre_id_num

    return rid_acc,rid_recall,rid_precision




def add_penalty_to_loss(logits, targets, penalty_matrix=None):
    """
    将惩罚矩阵应用到模型计算的交叉熵损失上

    参数:
    logits (Tensor): 模型输出的logits，形状为 (batch_size, vocab_size)
    targets (Tensor): 目标标签，形状为 (batch_size,)
    penalty_matrix (numpy.ndarray): 惩罚矩阵，形状为 (vocab_size, vocab_size)

    返回:
    penalties (Tensor): 惩罚值
    """
    

    
    # 过滤掉 targets 中为 -100 的部分
    mask = targets != -100
    filtered_logits = logits[mask]
    filtered_targets = targets[mask]

    # 计算 softmax 概率
    probabilities = F.softmax(filtered_logits, dim=1)

    # 使用 gather 方法选择 target 行的惩罚矩阵
    target_penalties = penalty_matrix[filtered_targets]

    # 计算每个样本的惩罚
    penalties = torch.sum( -torch.log(probabilities)* target_penalties, dim=1)

    # 返回总惩罚
    return torch.mean(penalties)


def add_penalty_to_los_2(logits, targets, penalty_matrix=None):
    """
    将惩罚矩阵应用到模型计算的交叉熵损失上

    参数:
    logits (Tensor): 模型输出的logits，形状为 (batch_size, vocab_size)
    targets (Tensor): 目标标签，形状为 (batch_size,)
    penalty_matrix (numpy.ndarray): 惩罚矩阵，形状为 (vocab_size, vocab_size)

    返回:
    penalties (Tensor): 惩罚值
    """
    
    # 过滤掉 targets 中为 -100 的部分
    mask = targets != -100
    filtered_logits = logits[mask]
    filtered_targets = targets[mask]

    # 计算 softmax 概率
    probabilities = F.softmax(filtered_logits, dim=1)

    # 使用 gather 方法选择 target 行的惩罚矩阵
    target_penalties = penalty_matrix[filtered_targets]

    penalties = torch.sum(-torch.log(probabilities) / torch.log(target_penalties), dim=1)


    # 返回总惩罚
    return torch.mean(penalties)

# 邻接矩阵loss 计算前一个 token 和预测的 token 如果联通则 loss 为 0 不联通 loss 为 1
def  adj_loss(logits, targets,adj_matrix = None,base=100):
    if adj_matrix == None:
        raise ValueError('adj_matrix is None')
    
    # logits shape (batch_size, seq_len, vocab_size)
    # 创建一个与logits最后一个时间步相同形状的全0填充张量
    padding = torch.zeros_like(logits[:, :1, :])  # 使用zeros_like来匹配形状并填充0
    #将logits 向前移动一位
    logits = logits[:,1:,:]
    #补充0
    # 将修改后的logits与新的填充张量拼接
    logits = torch.cat([logits, padding], dim=1)

    m_targets = targets[:,1:] 
    # 创建一个全是-100的张量，形状与targets最后一列相同
    pad = torch.full_like(targets[:, :1], -100)  # 使用full_like来匹配形状并填充-100

    # 将原始的targets（去掉第一个元素后）与新的-100填充列拼接
    m_targets = torch.cat([m_targets, pad], dim=1)

    # 过滤掉 targets 中为 -100 的部分
    mask = m_targets != -100
    filtered_targets = targets[mask] #shape (batch_size, seq_len)
    # 计算 softmax 概率
    probabilities = F.softmax(logits, dim=1) #shape ：(batch_size, seq_len, vocab_size)
    probabilities = probabilities[mask]
    # 使用 gather 方法选择 target 行的惩罚矩阵
    filtered_targets = adj_matrix[filtered_targets] # shape (batch_size, seq_len, vocab_size)  3528
    #计算 logits 作为幂，100 的幂次方结果
    probabilities = base ** probabilities
    
    #logits 和 matrix 乘积 计算loss
    loss = torch.sum(probabilities * filtered_targets,dim=1)
    loss = torch.mean(loss)

    return loss
    