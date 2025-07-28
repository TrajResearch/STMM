from collections import Counter
from transformers import PreTrainedTokenizer
import os
class CustomT5Tokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, pad_token="<pad>", eos_token="</s>", unk_token="<unk>",mask_token="<extra_id_0>", *args, **kwargs):
        self.vocab = self.load_vocab(vocab_file)
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.mask_token = mask_token
        # 先初始化ids_to_tokens
        self.ids_to_tokens = {id_: token for token, id_ in self.vocab.items()}

        # 确保必要的特殊令牌存在
        self.ensure_special_token_exists(self.pad_token)
        self.ensure_special_token_exists(self.eos_token)
        self.ensure_special_token_exists(self.unk_token)

        self.ensure_special_token_exists(self.mask_token)
        # self.ensure_special_token_exists("<extra_id_1>")

        # 更新self.ids_to_tokens以确保包含了所有新添加的特殊令牌
        self.ids_to_tokens = {id_: token for token, id_ in self.vocab.items()}
        super().__init__(pad_token=self.pad_token, eos_token=self.eos_token, unk_token=self.unk_token,mask_token=self.mask_token, *args, **kwargs)
    
    def ensure_special_token_exists(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
            self.ids_to_tokens[self.vocab[token]] = token

    def load_vocab(self, vocab_file):
        vocab = {}
        with open(vocab_file, 'r') as f:
            for index, line in enumerate(f.readlines()):
                vocab[line.strip()] = index
        return vocab

    def get_vocab(self):
        return self.vocab.copy()

    def _tokenize(self, text, **kwargs):
        return text.split()

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        tokens = []
        for id_ in ids:
            tokens.append(self.ids_to_tokens.get(id_, self.unk_token))
        return tokens

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 为序列添加EOS令牌
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        else:
            return token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
    
    @property
    def vocab_size(self):
        return len(self.vocab)
    
    def save_vocabulary(self, save_directory, filename_prefix=None):
        # 确定词汇表文件的路径
        if filename_prefix:
            vocab_file = os.path.join(save_directory, f"{filename_prefix}-vocab.txt")
        else:
            vocab_file = os.path.join(save_directory, "vocab.txt")

        # 将词汇表写入文件
        with open(vocab_file, 'w', encoding='utf-8') as writer:
            for token, index in sorted(self.vocab.items(), key=lambda item: item[1]):
                writer.write(token + '\n')

        # 返回词汇表文件的路径
        return (vocab_file,)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_args, **kwargs):
        # 假设词汇表文件名是固定的，比如 vocab.txt
        vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.txt")

        # 检查词汇表文件是否存在
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Vocab file {vocab_file} not found")

        # 现在我们有了 vocab_file，可以将其作为参数传递给类初始化器
        return cls(vocab_file, *init_args, **kwargs)


