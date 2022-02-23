import math
import os
import jieba
import collections
from IPython import display
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from foxrelax.ml import torch as ml

display.set_matplotlib_formats('svg')
"""
Transformer从零开始实现
"""


def tokenizer_cn(text):
    return list(jieba.cut(text, cut_all=False))


def count_corpus(tokens):
    """
    统计token的频率
    """
    if tokens and isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        参数:
        tokens: 2D list
        min_freq: 单个token的最小频率, 小于这个频率会被忽略
        reserved_tokens: 预留的token
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # 按出现频率排序
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(),
                                  key=lambda x: x[1],
                                  reverse=True)
        # 未知token <unk>的索引为0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [
            token for token, freq in self.token_freqs
            if freq >= min_freq and token not in uniq_tokens
        ]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """
        返回vocab中token的个数
        """
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """
        tokens -> int list

        例如:
        en_vocab[['keep', 'tom', 'safe']] -> [202, 12, 859]
        tgt_vocab[['确保', '汤姆', '安全']] -> [3935, 13, 499]
        """
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """
        int list -> tokens

        例如:
        en_vocab.to_tokens([202, 12, 859]) -> ['keep', 'tom', 'safe']
        cn_vocab.to_tokens([3935, 13, 499]) -> ['确保', '汤姆', '安全']
        """
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def read_data_nmt():
    """
    载入'英语－中文'数据集
    """
    data_dir = ml.download_extract('cmn_eng')
    with open(os.path.join(data_dir, 'cmn.txt'), 'r', encoding='utf-8') as f:
        return f.read()


def preprocess_nmt(text):
    """
    预处理'英语－中文'数据集

    做2个简单处理:
    1. 使用小写字母替换大写字母, 忽略英文大小写字母的差异
    2. 在英文单词和标点符号之间插入空格, 这样可以把标点符号作为一个单独的token切分出来
    """
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [
        ' ' + char if i > 0 and no_space(char, text[i - 1]) else char
        for i, char in enumerate(text)
    ]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """
    词元化'英语－中文'数据数据集

    一行数据当做一个样本, `num_examples`表示最大样本数, 
    也就是最多处理多少行数据

    返回的source和target是一个2D list, list的每个元素也是一个list, 表示一个训练样本(一句话), 格式如下:
    source[1000] = ['keep', 'tom', 'safe', '.']
    target[1000] = ['确保', '汤姆', '安全', '。']
    """
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 3:
            source.append(parts[0].split(' '))
            target.append(tokenizer_cn(parts[1]))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    """
    截断或填充文本序列
    """
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def build_array_nmt(lines, vocab, num_steps):
    """
    将机器翻译的文本序列转换成长度相同的line(截断或者填充), 这样返回的每
    个line都有相同的长度, 后续可以将其转换为小批量一起处理

    1. 在句子结尾增加<eos>
    2. 长度不足`num_steps`的, 用<pad>补足
    """
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor(
        [truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_array(data_arrays, batch_size, is_train=True):
    """
    构造一个PyTorch数据迭代器
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def load_data_nmt(batch_size, num_steps, num_examples=None):
    """
    返回翻译数据集的迭代器和词汇表

    英文-中文

    1. 28447条训练样本(每个样本一个词/一个短语/一句话)
    2. 构造的词典: 中文-7578个token; 英文-4526个token
    """
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source,
                      min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target,
                      min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


class Encoder(nn.Module):
    """
    Encoder-Decoder架构的基本Encoder接口
    """
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """
    Encoder-Decoder架构的基本Decoder接口
    """
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class AttentionDecoder(Decoder):
    """
    带有注意力机制的Decoder接口
    """
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """
    Encoder-Decoder架构的类
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        """
        参数:
        ffn_num_input: 输入的特征维度
        ffn_num_hiddens: 隐藏层的维度
        ffn_num_outputs: 输出的特征维度
        """
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        """
        参数:
        X的形状: (batch_size, num_steps, ffn_num_input)

        返回:
        output的形状: (batch_size, num_steps, ffn_num_outputs)
        """
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        """
        参数:
        normalized_shape: (num_steps, num_dims)
        dropout: dropout操作设置为0的元素的概率
        """
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        """
        将相同形状的X, Y按元素相加, 然后再做一次LayerNorm后输出,
        输出的形状和X, Y的形状是一样的

        参数:
        X的形状: (batch_size, num_steps, num_dims)
        Y的形状: (batch_size, num_steps, num_dims)

        返回:
        output的形状: (batch_size, num_steps, num_dims)
        """
        assert X.shape == Y.shape
        return self.ln(self.dropout(Y) + X)


def sequence_mask(X, valid_len, value=0):
    """
    在X序列中屏蔽不相关的项, 会把不相关的项替换成value

    参数:
    X的形状: (batch_size, num_steps)或者(batch_size, num_steps, num_dims)
    valid_len的形状: (batch_size, )

    返回:
    output的形状: (batch_size, num_steps)或者(batch_size, num_steps, num_dims)
    """
    maxlen = X.size(1)
    # mask经过广播后的形状是:
    # [1, num_steps] < [batch_size, 1] -> [batch_size, num_steps]
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """
    通过在最后一个维度(进行softmax计算的维度)遮蔽元素来执行softmax操作,
    被遮蔽的元素使用一个非常大的负值替换, 从而其softmax(指数)输出为0

    参数:    
    X的形状: (batch_size, num_steps, num_dims)
    valid_lens的形状: (batch_size, )或者(batch_size, num_steps)

    返回:
    output的形状: (batch_size, num_steps, num_dims)
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        # 将valid_lens转换成1D张量: (batch_size * num_steps, )
        if valid_lens.dim() == 1:
            # valid_lens是1D张量就是(batch_size, ) -> (batch_size * num_steps, )
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # valid_lens是2D张量就是(batch_size, num_steps) -> (batch_size * num_steps, )
            valid_lens = valid_lens.reshape(-1)

        # 在最后的轴上, 被遮蔽的元素使用一个非常大的负值替换, 从而其softmax(指数)输出为0
        # 将X转换成: (batch_size, num_steps, num_dims) -> (batch_size * num_steps, num_dims)
        # valid_lens: (batch_size * num_steps, )
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)

        # 返回的X形状: (batch_size, num_steps, num_dims)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """
    缩放点积注意力

    注意:
    1. 这种注意力实现当中是没有参数需要学习的(加性注意力有参数需要学习)
    2. valid_lens的作用是屏蔽部分values, 在计算输出的时候只会看valid_lens范围内的values,
       超出部分的values对应的attention_weights会被设置为0, 也就是忽略这部分values
    """
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        参数:
        queries的形状：(batch_size, num_query, d)
        keys的形状：(batch_size, num_kv, d)
        values的形状：(batch_size, num_kv, value_size)
        valid_lens的形状: (batch_size,) 或者(batch_size, num_query)

        返回:
        output的形状: (batch_size, num_query, value_size)
        """
        d = queries.shape[-1]
        # scores的形状：(batch_size, num_query, num_kv)
        # (batch_size, num_query, d) x (batch_size, d, num_kv) -> (batch_size, num_query, num_kv)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # attention_weights的形状：(batch_size, num_query, num_kv)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # (batch_size, num_query, num_kv) x (batch_size, num_kv, value_size) -> (batch_size, num_query, value_size)
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_qkv(X, num_heads):
    """
    输入:
    X的形状: (batch_size, num_query或者num_kv, num_hiddens)
    num_heads: head的个数

    返回:
    output的形状: (batch_size*num_heads, num_query或者num_kv, num_hiddens/num_heads)
    """
    # 输入X的形状: (batch_size, num_query或者num_kv, num_hiddens)
    # 变换后X的形状: (batch_size, num_query或者num_kv, num_heads, num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 变换后X的形状: (batch_size, num_heads, num_query或者num_kv, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # output的形状: (batch_size*num_heads, num_query或者num_kv, num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """
    逆转`transpose_qkv`函数的操作

    输入:
    X的形状: (batch_size*num_heads, num_query, num_hiddens/num_heads)

    返回:
    output的形状: (batch_size, num_query, num_hiddens)
    """
    # 输入X的形状: (batch_size*num_heads, num_query, num_hiddens/num_heads)
    # 变换后X的形状: (batch_size, num_heads, num_query, num_hiddens/num_heads)
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])

    #  变换后X的形状: (batch_size, num_query, num_heads, num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)
    # output的形状: (batch_size, num_query, num_hiddens)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 key_size,
                 query_size,
                 value_size,
                 num_hiddens,
                 num_heads,
                 dropout,
                 bias=False,
                 **kwargs):
        """
        参数:

        key_size: key的特征长度
        query_size: query的特征长度
        value_size: value的特征长度
        num_hiddens: 隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        num_heads: 头数
        dropout: dropout操作设置为0的元素的概率
        bias: 是否开启bias
        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads

        # 缩放点积注意力
        # 由于缩放点积注意力没有需要学习的参数, 所以可以让多个头共享1个attention
        self.attention = DotProductAttention(dropout)

        # 注意:
        # 每个头应该有一个单独的W_q, W_k, W_v, 在这里我们为了实现多个头的`并行计算`,
        # 将num_heads个头的W_q, W_k, W_v合并到一起, 这样多个头可以`并行计算`, 效率更高
        #
        # 举例说明:
        # 如果有8个头, 我们每个头会有24个矩阵:
        # W_q_1, W_q_2, ....W_q_8, 形状为: (query_size, num_hiddens/8)
        # W_k_1, W_k_2, ....W_k_8, 形状为: (key_size, num_hiddens/8)
        # W_v_1, W_v_2, ....W_v_8, 形状为: (value_size, num_hiddens/8)
        #
        # 当前的并行版本将8个头的24个矩阵合并为3个矩阵:
        # W_q, 形状为: (query_size, num_hiddens)
        # W_k, 形状为: (key_size, num_hiddens)
        # W_v, 形状为: (value_size, num_hiddens)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)

        # 最终输出层的线性变换矩阵
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """
        当前的版本是效率比较高的实现方式, 我们没有单独计算每个头, 
        而是通过变换, 并行的计算多个头

        参数:
        queries的形状: (batch_size, num_query, query_size)
        keys的形状: (batch_size, num_kv, key_size)
        values的形状: (batch_size, num_kv, value_size)
        valid_lens的形状: (batch_size, )或者(batch_size, num_query)

        返回:
        output的形状: (batch_size, num_query, num_hiddens)
        """
        # self.W_q(queries) -> (batch_size, num_query, num_hiddens)
        # self.W_k(keys) -> (batch_size, num_kv, num_hiddens)
        # self.W_v(values) -> (batch_size, num_kv, num_hiddens)
        #
        # 经过变换后
        # queries -> (batch_size * num_head, num_query, num_hiddens/num_heads)
        # keys -> (batch_size * num_head, num_kv, num_hiddens/num_heads)
        # values -> (batch_size * num_head, num_kv, num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0, 将第一项(标量或者矢量)复制`num_heads`次,
            # (batch_size*num_heads, )或者(batch_size*num_heads, num_query)
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads,
                                                 dim=0)

        # output的形状: (batch_size * num_heads, num_query, num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状: (batch_size, num_query, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)

        # output的形状: (batch_size, num_query, num_hiddens)
        return self.W_o(output_concat)


class PositionalEncoding(nn.Module):
    """
    实现位置编码
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
        """
        参数:
        num_hiddens: 特征长度
        dropout: dropout操作设置为0的元素的概率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的`P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(
                10000,
                torch.arange(0, num_hiddens, 2, dtype=torch.float32) /
                num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        """
        输入:
        X的形状: (batch_size, num_steps, num_hiddens)

        返回:
        output的形状: (batch_size, num_steps, num_hiddens)
        """
        # X的形状: (batch_size, num_steps, num_hiddens)
        # P的形状: (1, max_len, num_hiddens)
        # 在相加的时候, P在第一个维度可以通过广播来进行计算
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class EncoderBlock(nn.Module):
    def __init__(self,
                 key_size,
                 query_size,
                 value_size,
                 num_hiddens,
                 norm_shape,
                 ffn_num_input,
                 ffn_num_hiddens,
                 num_heads,
                 dropout,
                 use_bias=False,
                 **kwargs):
        """
        参数:
        key_size: key的特征长度
        query_size: query的特征长度
        value_size: value的特征长度
        num_hiddens: 隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        norm_shape: (num_steps, num_hiddens)
        ffn_num_input: ffn输入的维度
        ffn_num_hiddens: ffn隐藏层的维度
        num_heads: 头数
        dropout: dropout操作设置为0的元素的概率
        use_bias: 是否开启bias
        """
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size,
                                            num_hiddens, num_heads, dropout,
                                            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        """
        参数:
        X的形状: (batch_size, num_steps, num_hiddens)
        valid_lens的形状: (batch_size, ), 表示X对应的有效token个数

        返回:
        output的形状: (batch_size, num_steps, num_hiddens)
        """
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(Encoder):
    def __init__(self,
                 vocab_size,
                 key_size,
                 query_size,
                 value_size,
                 num_hiddens,
                 norm_shape,
                 ffn_num_input,
                 ffn_num_hiddens,
                 num_heads,
                 num_layers,
                 dropout,
                 use_bias=False,
                 **kwargs):
        """
        参数:
        vocab_size: 字典大小
        key_size: key的特征长度
        query_size: query的特征长度
        value_size: value的特征长度
        num_hiddens: 隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        norm_shape: (num_steps, num_hiddens)
        ffn_num_input: ffn输入的维度
        ffn_num_hiddens: ffn隐藏层的维度
        num_heads: 头数
        num_layers: EncoderBlock的数量
        dropout: dropout操作设置为0的元素的概率
        use_bias: 是否开启bias
        """
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        """
        输入:
        X的形状: (batch_size, num_steps)
        valid_lens的形状: (batch_size, ), 表示X对应的有效token个数

        返回:
        output的形状: (batch_size, num_steps, num_hiddens)
        """
        # 因为位置编码值在-1和1之间, 因此嵌入值乘以嵌入维度的平方根进行缩放,
        # 然后再与位置编码相加
        #
        # X的形状: (batch_size, num_steps)
        # 处理后的X的形状: (batch_size, num_steps, num_hiddens)
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))

        # 每一个block记录一个attention_weight
        # 每个的attention_weight的形状是根据送入attention时的queries, keys, values的形状
        # 计算出来的: (batch_size, num_query, num_kv)
        self.attention_weights = [None] * len(self.blks)

        # 遍历每个EncoderBlock
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            # 记录下attention_weights, 后续可以显示
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    """
    解码器中第i个块
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        """
        参数:
        key_size: key的特征长度
        query_size: query的特征长度
        value_size: value的特征长度
        num_hiddens: 隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        norm_shape: (num_steps, num_hiddens)
        ffn_num_input: ffn输入的维度
        ffn_num_hiddens: ffn隐藏层的维度
        num_heads: 头数
        dropout: dropout操作设置为0的元素的概率
        i: 第几个DecoderBlock
        """
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size,
                                             num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size,
                                             num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        """
        参数:
        X的形状: (batch_size, num_steps, num_hiddens)
        state的形状: [enc_outputs, enc_valid_lens, list]
                    enc_outputs的形状 - (batch_size, num_steps, num_hiddens)
                    enc_valid_lens的形状 - (batch_size, )

        返回: (output, state)
        output的形状: (batch_size, num_steps, num_hiddens)
        """
        # enc_outputs的形状: (batch_size, num_steps, num_hiddens)
        # enc_valid_lens的形状: (batch_size, )
        enc_outputs, enc_valid_lens = state[0], state[1]

        # 1. 训练阶段, 输出序列的所有token都在同一时间处理,
        #    因此state[2][self.i]初始化为None.
        # 2. 预测阶段, 输出序列是通过token一个接着一个解码的, 因此state[2][self.i]包含
        #    着直到当前时间步第i个DecoderBlock解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的形状: (batch_size, num_steps)
            #
            # 例如: (batch_size=4, num_steps=6)
            # tensor([[1, 2, 3, 4, 5, 6],
            #         [1, 2, 3, 4, 5, 6],
            #         [1, 2, 3, 4, 5, 6],
            #         [1, 2, 3, 4, 5, 6]])
            dec_valid_lens = torch.arange(1, num_steps + 1,
                                          device=X.device).repeat(
                                              batch_size, 1)
        else:
            dec_valid_lens = None

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)

        # 编码器－解码器注意力
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        """
        vocab_size: 字典大小
        key_size: key的特征长度
        query_size: query的特征长度
        value_size: value的特征长度
        num_hiddens: 隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        norm_shape: (num_steps, num_hiddens)
        ffn_num_input: ffn输入的维度
        ffn_num_hiddens: ffn隐藏层的维度
        num_heads: 头数
        num_layers: DecoderBlock的数量
        dropout: dropout操作设置为0的元素的概率
        """
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)  # 最终的输出层

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        """
        参数:
        enc_outputs的形状: (batch_size, num_steps, num_hiddens)
        enc_valid_lens的形状: (batch_size, )
        """
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        """
        参数:
        X的形状: (batch_size, num_steps)
        state的形状: [enc_outputs, enc_valid_lens, [None] * self.num_layers]
                    enc_outputs的形状 - (batch_size, num_steps, num_hiddens)
                    enc_valid_lens的形状 - (batch_size, )
        返回: (output, state)
        output的形状: (batch_size, num_steps, vocab_size)
        """
        # 处理后的X的形状: (batch_size, num_steps, num_hiddens)
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            # X的形状: (batch_size, num_steps, num_hiddens)
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # '编码器－解码器'自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights

        # 经过一个线性层之后输出最终结果
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


def grad_clipping(net, theta):
    """
    裁剪梯度

    确保梯度的L2 Norm不超过theta
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """
    带遮蔽的softmax交叉熵损失函数

    计算过程如下:
    假设pred的形状为: (2, 5, 10), label的形状为: (2, 5), 则reduction=none时, 计算出来
    的loss的形状为(2, 5), 如下:
    tensor([[2.4712, 1.7931, 1.6518, 2.3004, 1.0466],
            [3.5565, 2.1062, 3.2549, 3.9885, 2.7302]])

    我们叠加如下的valid_len=tensor([5, 2]), 则会生成如下weights
    tensor([[1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0]])

    所以最终计算出来的loss为:
    tensor([[2.4712, 1.7931, 1.6518, 2.3004, 1.0466],
            [3.5565, 2.1062, 0, 0, 0]])
    最终得到的loss为: tensor([1.8526, 1.1325])
    (2.4712+1.7931+1.6518+2.3004+1.0466)/5 = 1.8526
    (3.5565+2.1062)/5 = 1.1325
    """
    def forward(self, pred, label, valid_len):
        """
        参数:
        pred的形状: (batch_size, num_steps, vocab_size)
        label的形状: (batch_size, num_steps)
        valid_len的形状: (batch_size,)
        """
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        # unweighted_loss的形状: (batch_size, num_steps)
        # 下面要做permute操作是因为我们需要把C这个维度放到第二个维度上, 这是框架的要求
        unweighted_loss = super(MaskedSoftmaxCELoss,
                                self).forward(pred.permute(0, 2, 1), label)
        # weighted_loss的形状: (batch_size,)
        weighted_loss = (weights * unweighted_loss).mean(dim=1)
        return weighted_loss


# 训练:
# num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 16
# lr, num_epochs, device = 0.005, 200, ml.try_gpu()
# ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
# key_size, query_size, value_size = 32, 32, 32
# norm_shape = [32]

# train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)

# encoder = TransformerEncoder(len(src_vocab), key_size, query_size, value_size,
#                              num_hiddens, norm_shape, ffn_num_input,
#                              ffn_num_hiddens, num_heads, num_layers, dropout)
# decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size,
#                              num_hiddens, norm_shape, ffn_num_input,
#                              ffn_num_hiddens, num_heads, num_layers, dropout)
# net = EncoderDecoder(encoder, decoder)
#
# train_seq2seq_gpu(net, train_iter, lr, num_epochs, tgt_vocab, device)


def train_seq2seq_gpu(net, data_iter, lr, num_epochs, tgt_vocab, device=None):
    """
    训练序列到序列模型
    """
    if device is None:
        device = ml.try_gpu()

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
        elif type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if 'weight' in param:
                    nn.init.xavier_normal_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = ml.Animator(xlabel='epoch',
                           ylabel='loss',
                           xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = ml.Timer()
        metric = ml.Accumulator(2)  # 训练损失总和, 词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            # X的形状: (batch_size, num_steps)
            # X_valid_len的形状: (batch_size, )
            # Y的形状: (batch_size, num_steps)
            # Y_valid_len的形状: (batch_size, )
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # bos的形状: (batch_size, 1)
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)

            # dec_input的形状: (batch_size, num_steps)
            # 每个批量的第一个时间步输入的是<bos>, 比如原始的Y是[[w1, w2, w3, w4, w5],...]
            # 经过处理后的dec_input是: [[<bos>, w1, w2, w3, w4], ...], 所有的token左移一个位置,
            # 并删除了最后一个token
            dec_input = torch.cat((bos, Y[:, :-1]), 1)
            # Y_hat的形状: (batch_size, num_steps, vocab_size)
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # 损失函数的标量进行'反传'
            grad_clipping(net, 1)  # 梯度裁剪
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
            if (epoch + 1) % 10 == 0:
                animator.add(epoch + 1, (metric[0] / metric[1], ))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')


def bleu(pred_seq, label_seq, k):
    """
    计算BLEU

    pred_seq, label_seq都是中文的句子
    """
    pred_tokens, label_tokens = tokenizer_cn(pred_seq), tokenizer_cn(label_seq)
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i:i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i:i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i:i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def predict_seq2seq(net,
                    src_setence,
                    src_vocab,
                    tgt_vocab,
                    num_steps,
                    device,
                    save_attention_weights=False):
    """
    序列到序列模型的预测(英文 -> 中文)

    预测结束条件: 最多预测num_steps步或者预测到<eos>结束
    """
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_setence.lower().split(' ')] + [
        src_vocab['<eos>']
    ]
    # enc_valid_len的形状是: (1, )
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加batch_size轴
    # enc_X的形状是: (1, num_steps)
    enc_X = torch.unsqueeze(torch.tensor(src_tokens,
                                         dtype=torch.long,
                                         device=device),
                            dim=0)

    # 对输入的enc_X进行编码(提取特征):
    # enc_outputs的形状是: (1, num_steps, num_hiddens)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加batch_size轴
    # dec_X的形状是: (1, 1)
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']],
                                         dtype=torch.long,
                                         device=device),
                            dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        # Y的形状: (1, 1, vocab_size)
        Y, dec_state = net.decoder(dec_X, dec_state)
        # dec_X的形状: (1, 1)
        # 我们使用具有预测最高可能性的token, 作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重, 每一个step都会保存一个
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束token被预测, 输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    print(tgt_vocab.to_tokens(output_seq))
    return ''.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


# 预测:
# ens = [
#     'Where is your brother ?', "All of these books are mine .",
#     "He gave me a nice Christmas present ."
# ]
# cns = ['你的哥哥在哪里?', '这些书都是我的。', '他给了我一个很棒的圣诞礼物。']
# for en, cn in zip(ens, cns):
#     translation, dec_attention_weight_seq = predict_seq2seq(
#         net, en, src_vocab, tgt_vocab, num_steps, device, True)
#     print(f'{en} => {translation}, ', f'bleu {bleu(translation, cn, k=2):.3f}')
# ['你', '哥哥', '在', '哪裡', '？']
# Where is your brother ? => 你哥哥在哪裡？,  bleu 0.448
# ['這些', '書', '是', '我', '的', '書', '。']
# All of these books are mine . => 這些書是我的書。,  bleu 0.574
# ['他給', '了', '我', '一個', '不錯', '的', '聖', '誕禮物', '。']
# He gave me a nice Christmas present . => 他給了我一個不錯的聖誕禮物。,  bleu 0.396