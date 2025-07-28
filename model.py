

from transformers import T5Config, T5ForConditionalGeneration
from transformers import T5EncoderModel
from torch.nn import Linear
from torch import nn
import torch
import os

class T5MLM(nn.Module):
    def __init__(self, model_path, tokenizer, device,model_conf):
        super(T5MLM, self).__init__()
        if model_path is None:
            config = T5Config(
                vocab_size=tokenizer.vocab_size,
                d_model = model_conf.hidden_size,
                d_kv=model_conf.d_kv,
                d_ff=model_conf.d_ff,
                num_layers=model_conf.num_layers,
                num_heads=model_conf.num_heads,
                relative_attention_num_buckets=model_conf.relative_attention_num_buckets,
                dropout_rate=0.1,
                layer_norm_epsilon=1e-6,
                initializer_factor=1.0,
                pad_token_id=tokenizer.pad_token_id,
                num_decoder_layers=0,  
                mask_token_id = tokenizer.mask_token_id,
                is_encoder_decoder=True,
            )
            self.model = T5EncoderModel(config)
        else:
            self.model = T5EncoderModel.from_pretrained(model_path)
        self.model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.config = config    
        # 输出层，将输出投射回词汇空间
        self.mlm_layer = Linear(model_conf.hidden_size, tokenizer.vocab_size)
        # 可选地绑定权重，如同Tied Word Embedding
        self.mlm_layer.weight = self.model.shared.weight
        self.mlm_layer.to(device)

        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        mlm_outputs = self.mlm_layer(outputs.last_hidden_state)

        loss = self.loss_fct(mlm_outputs.view(-1, self.tokenizer.vocab_size), labels.view(-1))
        outputs.loss = loss
        return outputs
    
    def save_model(self, encoder_path, mlm_layer_path,t5_encoder_path):
        torch.save(self.model.state_dict(), encoder_path)
        torch.save(self.mlm_layer.state_dict(), mlm_layer_path)
        self.model.save_pretrained(t5_encoder_path) #保存t5 encoder模型
    
    @classmethod
    def load_model(cls, encoder_path, mlm_layer_path,tokenizer, device):
        instance = cls(None, tokenizer, device)  # 初始化一个新实例

        # 加载T5EncoderModel
        encoder_model_state = torch.load(encoder_path, map_location=device)
        instance.model.load_state_dict(encoder_model_state)

        # 加载MLM层
        mlm_layer_state = torch.load(mlm_layer_path, map_location=device)
        instance.mlm_layer.load_state_dict(mlm_layer_state)
        
         
        return instance

        
class T5Generator(nn.Module):
    def __init__(self, model_path, tokenizer, device,model_conf, if_init_weight=False):
        super(T5Generator, self).__init__()
        if model_path is None:
            config = T5Config(
                vocab_size=tokenizer.vocab_size,
                d_model = model_conf.hidden_size,
                d_kv=model_conf.d_kv,
                d_ff=model_conf.d_ff,
                num_layers=model_conf.num_layers,
                num_heads=model_conf.num_heads,
                relative_attention_num_buckets=model_conf.relative_attention_num_buckets,
                num_decoder_layers=model_conf.num_decoder_layers,
                dropout_rate=0.1,
                layer_norm_epsilon=1e-6,
                initializer_factor=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                mask_token_id = tokenizer.mask_token_id,
                is_encoder_decoder=True,
                decoder_start_token_id=tokenizer.pad_token_id
            )
            self.model = T5ForConditionalGeneration(config)
        else:
            self.model_conf = model_conf
            self.load_encoder_weights(model_path,tokenizer)
            if_init_weight = True #如果有 则初始化 decoder
        self.device = device
        if if_init_weight:
            self.init_weight()
        
        self.model.to(device)

    def forward(self, input_ids, attention_mask, decoder_input_ids, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
        return outputs

    def init_weight(self):
        print('Initiating decoder attention weight')
        for encoder_layer, decoder_layer in zip(self.model.encoder.block, self.model.decoder.block):
            # 复制自注意力层权重
            decoder_layer.layer[0].SelfAttention.q.weight.data = encoder_layer.layer[0].SelfAttention.q.weight.data.clone()
            decoder_layer.layer[0].SelfAttention.k.weight.data = encoder_layer.layer[0].SelfAttention.k.weight.data.clone()
            decoder_layer.layer[0].SelfAttention.v.weight.data = encoder_layer.layer[0].SelfAttention.v.weight.data.clone()
            decoder_layer.layer[0].SelfAttention.o.weight.data = encoder_layer.layer[0].SelfAttention.o.weight.data.clone()

            if encoder_layer.layer[0].SelfAttention.q.bias is not None:
                decoder_layer.layer[0].SelfAttention.q.bias.data = encoder_layer.layer[0].SelfAttention.q.bias.data.clone()
                decoder_layer.layer[0].SelfAttention.k.bias.data = encoder_layer.layer[0].SelfAttention.k.bias.data.clone()
                decoder_layer.layer[0].SelfAttention.v.bias.data = encoder_layer.layer[0].SelfAttention.v.bias.data.clone()
                decoder_layer.layer[0].SelfAttention.o.bias.data = encoder_layer.layer[0].SelfAttention.o.bias.data.clone()
                
    def save_pretrained(self, save_directory):
        # 确保保存路径存在
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        # 保存整个 T5 模型
        model_path = os.path.join(save_directory, "t5_generator_model.bin")
        torch.save(self.model.state_dict(), model_path)

    @classmethod
    def from_pretrained(cls, load_directory, tokenizer, device, if_init_weight=False):
        # 创建一个新的实例
        instance = cls(None, tokenizer, device, if_init_weight)
        # 加载模型
        model_path = os.path.join(load_directory, "t5_generator_model.bin")
        instance.model.load_state_dict(torch.load(model_path, map_location=device))
        return instance
    
    def load_encoder_weights(self, encoder_model_path,tokenizer):
        print('load encoder weights')
        config = T5Config(
                vocab_size=tokenizer.vocab_size,
                d_model = self.model_conf.hidden_size,
                d_kv= self.model_conf.d_kv,
                d_ff= self.model_conf.d_ff,
                num_layers= self.model_conf.num_layers,
                num_heads= self.model_conf.num_heads,
                relative_attention_num_buckets= self.model_conf.relative_attention_num_buckets,
                num_decoder_layers= self.model_conf.num_decoder_layers,
                dropout_rate=0.1,
                layer_norm_epsilon=1e-6,
                initializer_factor=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                mask_token_id = tokenizer.mask_token_id,
                is_encoder_decoder=True,
                decoder_start_token_id=tokenizer.pad_token_id
            )
        # 初始化T5序列到序列模型
        self.model = T5ForConditionalGeneration(config)
        
        # 加载T5编码器模型的权重
        encoder_model = T5EncoderModel.from_pretrained(encoder_model_path)
        # 准备一个新的状态字典，将`encoder.`前缀从键中移除
        new_state_dict = {}
        for key, value in encoder_model.state_dict().items():
            new_key = key.replace("encoder.", "")  # 移除前缀
            new_state_dict[new_key] = value
        
        # 使用更新后的状态字典加载权重
        missing_keys, unexpected_keys = self.model.encoder.load_state_dict(new_state_dict, strict=False)


class T5Discriminator(nn.Module):
    def __init__(self, model_path, tokenizer, device,model_conf):
        super(T5Discriminator, self).__init__()
        if model_path is None:
            config = T5Config(
                vocab_size=tokenizer.vocab_size,
                d_model = model_conf.hidden_size,
                d_kv=model_conf.d_kv,
                d_ff=model_conf.d_ff,
                num_layers=model_conf.num_layers,
                num_heads=model_conf.num_heads,
                relative_attention_num_buckets=model_conf.relative_attention_num_buckets,
                num_decoder_layers=model_conf.num_decoder_layers,
                dropout_rate=0.1,
                layer_norm_epsilon=1e-6,
                initializer_factor=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                mask_token_id = tokenizer.mask_token_id,
                is_encoder_decoder=True,
                decoder_start_token_id=tokenizer.pad_token_id
            )
            self.discriminator = T5EncoderModel(config)
        else:
            self.discriminator = T5EncoderModel.from_pretrained(model_path)
        self.discriminator.to(device)
        self.cls_layer = nn.Linear(model_conf.hidden_size, 2)
        self.cls_layer.to(device)

        self.loss_fct = nn.BCEWithLogitsLoss() 

    def forward(self, input_ids, attention_mask, labels=None):
        
        outputs = self.discriminator(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state.mean(dim=1, keepdim=True).squeeze(1) # mean pooling

        cls_logits = self.cls_layer(last_hidden_state)
        outputs.logits = cls_logits
        if labels is not None:
            fake_logits = cls_logits[:, 1].squeeze()#.squeeze()

            loss = self.loss_fct(fake_logits, labels.float())
            outputs.loss = loss
            
        return outputs
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        # Save the discriminator model
        discriminator_model_path = os.path.join(save_directory, "discriminator_model.bin")
        torch.save(self.discriminator.state_dict(), discriminator_model_path)

        # Save the classifier layer
        cls_layer_path = os.path.join(save_directory, "cls_layer.bin")
        torch.save(self.cls_layer.state_dict(), cls_layer_path)

    @classmethod
    def from_pretrained(cls, load_directory, tokenizer, device, hidden_size=256):
        # Initialize an instance of the class without loading weights
        instance = cls(None, tokenizer, device, hidden_size)

        # Load the discriminator model
        discriminator_model_path = os.path.join(load_directory, "discriminator_model.bin")
        instance.discriminator.load_state_dict(torch.load(discriminator_model_path, map_location=device))

        # Load the classifier layer
        cls_layer_path = os.path.join(load_directory, "cls_layer.bin")
        instance.cls_layer.load_state_dict(torch.load(cls_layer_path, map_location=device))

        return instance


class T5Gan(nn.Module):
    def __init__(self, model_path, discriminator, tokenizer, device,model_conf, max_length):
        super(T5Gan, self).__init__()
        self.generator = T5Generator(model_path, tokenizer, device,model_conf,if_init_weight=False)
        self.discriminator = discriminator
        # 冻结 T5Discriminator 参数
        for param in self.discriminator.parameters():
            param.requires_grad = False
        self.device = device
        self.d_criterion = nn.BCEWithLogitsLoss()
        self.use_generate_output = True # 使用这个时，要先将首个 pad 删掉，后面补充 pad
        self.max_length = max_length

    def forward(self, input_ids, attention_mask, decoder_input_ids, labels, use_discriminator=False):
        outputs = self.generator(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
        gen_loss = outputs.loss

        if use_discriminator:
            if self.use_generate_output:
                fake_input = self.get_generate_output(input_ids, attention_mask)
            else:
                fake_input = torch.argmax(outputs.logits, dim=-1)

            self.discriminator.eval()
            dis_outputs = self.discriminator(fake_input, attention_mask)
            fake_logits = dis_outputs.logits[:, 1].squeeze()  # 假设第二列对应于“生成文本”的logits
            # 生成器的目标是让判别器将其输出判定为真实的（即标签为1）
            gen_dis_loss = self.d_criterion(fake_logits, torch.ones(fake_logits.size(), device=self.device))

            # 总损失是生成器损失和判别器对生成器输出的判断损失
            outputs.loss = outputs.loss + gen_dis_loss
            self.discriminator.train()
            return outputs
        else:
            return outputs
            

    def get_generate_output(self, input_ids, attention_mask):
        # 生成文本
        self.generator.model.eval()
        generated_outputs = self.generator.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_length,
            top_k=2,
            do_sample=True,
            temperature=0.01,
        )
        # 生成的序列可能有不同的长度，因此需要进行适当的填充处理
        padded_outputs = torch.full((input_ids.size(0), self.max_length), fill_value=self.generator.model.config.pad_token_id, device=self.device)
        generated_outputs = generated_outputs[:, 1:]
        for i, output in enumerate(generated_outputs):
            length = min(output.size(0), self.max_length)
            padded_outputs[i, :length] = output[:length]

        self.generator.model.train()
        return padded_outputs
    
    def save_pretrained(self, save_directory):
        # 确保保存路径存在
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存生成器
        generator_path = os.path.join(save_directory, 'generator')
        self.generator.save_pretrained(generator_path)
        
        # 保存判别器
        discriminator_path = os.path.join(save_directory, 'discriminator')
        self.discriminator.save_pretrained(discriminator_path)

    @classmethod
    def from_pretrained(cls, load_directory,  tokenizer, device, max_length):
        # 加载生成器
        generator_path = os.path.join(load_directory, 'generator')
        generator = T5Generator.from_pretrained(generator_path, tokenizer, device)

        discriminator_path = os.path.join(load_directory, 'discriminator')
        # 加载判别器
        discriminator = T5Discriminator.from_pretrained(discriminator_path, tokenizer, device)

        # 初始化 T5Gan 实例
        instance = cls(None, discriminator, tokenizer, device, max_length)
        instance.generator = generator
        instance.discriminator = discriminator

        # 再次冻结判别器的参数
        for param in instance.discriminator.parameters():
            param.requires_grad = False

        return instance

