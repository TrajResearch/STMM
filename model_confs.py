
class model_confs():
    def __init__(self,hidden_size,d_kv,d_ff,num_layers,num_heads,relative_attention_num_buckets,num_decoder_layers):
        self.hidden_size = hidden_size
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers =  num_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.num_decoder_layers = num_decoder_layers