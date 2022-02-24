import os
import random
import collections
from IPython import display
import torch
from torch import nn
from foxrelax.ml import transformer
import foxrelax.ml.torch as ml

display.set_matplotlib_formats('svg')
"""
BERT从零开始实现
"""


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """
    将tokens_a和tokens_b拼接起来, 返回拼接后的tokens及其segments

    >>> tokens_a = ['this', 'movie', 'is', 'great']
    >>> tokens_b = ['i', 'like', 'it']
    >>> tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
    >>> tokens
    ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    >>> segments
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    """
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


class BERTEncoder(nn.Module):
    """
    BERT编码器

    本质就是改进版本的Transformer Encoder
    """
    def __init__(self,
                 vocab_size,
                 num_hiddens,
                 norm_shape,
                 ffn_num_input,
                 ffn_num_hiddens,
                 num_heads,
                 num_layers,
                 dropout,
                 max_len=1000,
                 key_size=768,
                 query_size=768,
                 value_size=768,
                 **kwargs):
        """
        参数:
        vocab_size: 字典大小
        num_hiddens: Transformer EncoderBlock隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        norm_shape: (num_steps, num_hiddens)
        ffn_num_input: ffn输入的维度
        ffn_num_hiddens: ffn隐藏层的维度
        num_heads: 头数
        num_layers: EncoderBlock的数量
        dropout: dropout操作设置为0的元素的概率
        max_len: Pos Embedding生成的向量的最大长度
        key_size: key的特征长度
        query_size: query的特征长度
        value_size: value的特征长度
        """
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f"{i}",
                transformer.EncoderBlock(key_size, query_size, value_size,
                                         num_hiddens, norm_shape,
                                         ffn_num_input, ffn_num_hiddens,
                                         num_heads, dropout, True))

        # 在BERT中, 位置嵌入是可学习的, 因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        """
        参数:
        tokens的形状: (batch_size, num_steps)
        segments的形状: (batch_size, num_steps)
        valid_lens的形状: (batch_size, ), 表示tokens对应的有效token个数

        返回:
        output的形状: (batch_size, num_steps, num_hiddens)
        """

        # X的形状: (batch_size, num_steps, num_hiddens)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


class MaskLM(nn.Module):
    """
    BERT的Masked Language Modeling
    """
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        """
        参数:
        vocab_size: 字典大小
        num_hiddens: 隐藏层大小
        num_inputs: 输入的维度
        """
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens), nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        """
        做2两件事情:
        1. 把X中pred_positions对应的特征抽取出来: (batch_size, num_steps, num_hiddens)
        2. 将抽取出来的特征送入MLP, 处理成(batch_size, num_pred, vocab_size)

        参数:
        X的形状: (batch_size, num_steps, num_hiddens)
        pred_positions的形状: (batch_size, num_pred)

        返回:
        output的形状: (batch_size, num_pred, vocab_size)
        """
        num_pred_positions = pred_positions.shape[1]  # num_pred
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设: batch_size=2，num_pred_positions=3
        # 则: batch_idx = tensor([0, 0, 0, 1, 1, 1])
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)

        # batch_idx的形状: (batch_size*num_pred, )
        # pred_positions的形状: (batch_size*num_pred, )
        # masked_X的形状: (batch_size*num_pred, num_hiddens)
        masked_X = X[batch_idx, pred_positions]
        # masked_X的形状: (batch_size, num_pred, num_hiddens)
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        # mlm_Y_hat的形状: (batch_size, num_pred, vocab_size)
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    """
    BERT的下一句预测任务
    """
    def __init__(self, num_inputs, **kwargs):
        """
        参数:
        num_inputs: 输入的维度
        """
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        """
        参数:
        X的形状: (batch_size, num_inputs)

        返回:
        output的形状: (batch_size, 2)
        """
        return self.output(X)


class BERTModel(nn.Module):
    """
    BERT模型
    """
    def __init__(self,
                 vocab_size,
                 num_hiddens,
                 norm_shape,
                 ffn_num_input,
                 ffn_num_hiddens,
                 num_heads,
                 num_layers,
                 dropout,
                 max_len=1000,
                 key_size=768,
                 query_size=768,
                 value_size=768,
                 hid_in_features=768,
                 mlm_in_features=768,
                 nsp_in_features=768):
        """
        参数:
        vocab_size: 字典大小
        num_hiddens: Transformer EncoderBlock隐藏单元的特征长度(也是多头注意力最终输出的特征长度)
        norm_shape: (num_steps, num_hiddens)
        ffn_num_input: ffn输入的维度
        ffn_num_hiddens: ffn隐藏层的维度
        num_heads: 头数
        num_layers: EncoderBlock的数量
        dropout: dropout操作设置为0的元素的概率
        max_len: Pos Embedding生成的向量的最大长度
        key_size: key的特征长度
        query_size: query的特征长度
        value_size: value的特征长度
        hid_in_features: hidden层输入的维度
        mlm_in_features: MLM输入的维度
        nsp_in_features: NSP输入的维度
        """
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size,
                                   num_hiddens,
                                   norm_shape,
                                   ffn_num_input,
                                   ffn_num_hiddens,
                                   num_heads,
                                   num_layers,
                                   dropout,
                                   max_len=max_len,
                                   key_size=key_size,
                                   query_size=query_size,
                                   value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        """
        逻辑:
        1. 将tokens送入BERTEncoder提取特征, 得到encoded_X: (batch_size, num_steps, num_hiddens)
        2. 将提取出来的特征encoded_X和pred_positions送入MLM, 得到MLM的输出: (batch_size, num_pred, vocab_size)
        3. 将提取出来的特征encoded_X[:, 0, :])形状为(batch_size, num_hiddens), 也就是每个句子的<cls>, 送入NSP, 得到NSP的输出: (batch_size, 2)

        参数:
        tokens的形状: (batch_size, num_steps)
        segments的形状: (batch_size, num_steps)
        valid_lens的形状: (batch_size, ), 表示tokens对应的有效token个数
        pred_positions的形状: (batch_size, num_pred)

        返回: encoded_X, mlm_Y_hat, nsp_Y_hat
        encoded_X的形状: (batch_size, num_steps, num_hiddens)
        mlm_Y_hat的形状: (batch_size, num_pred, vocab_size)
        nsp_Y_hat的形状: (batch_size, 2)
        """

        # encoded_X的形状: (batch_size, num_steps, num_hiddens)
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            # mlm_Y_hat的形状: (batch_size, num_pred, vocab_size)
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None

        # 用于下一句预测的多层感知机分类器的隐藏层, 0是'<cls>'标记的索引
        # 将(batch_size, num_hiddens)的数据送入hidden -> (batch_size, num_hiddens)
        # 将(batch_size, num_hiddens)的数据送入nsp -> (batch_size, 2)
        # nsp_Y_hat的形状: (batch_size, 2)
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


def tokenize(lines, token='word'):
    assert token in ('word', 'char'), 'Unknown token type: ' + token
    return [line.split() if token == 'word' else list(line) for line in lines]


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

        e.g. vocab[['keep', 'tom', 'safe']] -> [202, 12, 859]
        """
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """
        int list -> tokens

        e.g. vocab.to_tokens([202, 12, 859]) -> ['keep', 'tom', 'safe']
        """
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def _read_wiki(data_dir):
    """
    返回的paragraphs是一个paragraph列表, 每个paragraph包含多个sentence, 
    每个sentence包含多个token
    """
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # 大写字母转换为小写字母
    paragraphs = [
        line.strip().lower().split(' . ') for line in lines
        if len(line.split(' . ')) >= 2
    ]
    random.shuffle(paragraphs)  # 随机打乱段落的顺序
    return paragraphs


def _get_next_sentence(sentence, next_sentence, paragraphs):
    """
    输入: 
    sentence: e.g. ['this', 'movie', 'is', 'great']
    next_sentence: e.g. ['i', 'like', 'it']
    paragraphs: list of paragraph

    返回: (sentence, next_sentence, is_next)
    sentence: e.g. ['this', 'movie', 'is', 'great']
    next_sentence: e.g. ['i', 'like', 'it']
    is_next: True | False
    """
    if random.random() < 0.5:
        is_next = True
    else:
        # 先随机选择一个paragraph, 在从这个paragraph中随机选择一个sentence
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    """
    处理一个paragraph, 返回训练NSP的数据

    参数:
    paragraph: 句子列表, 其中每个句子都是token列表
               e.g. [['this', 'movie', 'is', 'great'], ['i', 'like', 'it']]
    paragraphs: list of paragraph
    vocab: 字典
    max_len: 预训练期间的BERT输入序列的最大长度(超过最大长度的tokens忽略掉)

    返回: list of (tokens, segments, is_next)
    tokens: ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    segments: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    is_next: True | False
    """
    nsp_data_from_paragraph = []  # [(tokens, segments, is_next)]
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 考虑1个'<cls>'和2个'<sep>', 超过最大长度max_len的tokens忽略掉
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    """
    处理一句话(tokens), 返回`MLM的输入, 预测位置以及标签`

    参数:
    tokens: 表示BERT输入序列的token列表
            e.g. ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    candidate_pred_positions: 后选预测位置的索引, 会在tokens中过滤掉<cls>, <sep>, 剩下的都算后选预测位置
                              (特殊token <cls>, <sep>在MLM任务中不被预测)
                              e.g. ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']对应的
                              后选预测位置是: [1,2,3,4,6,7,8]
    num_mlm_preds: 需要预测多少个token, 通常是len(tokens)的15%
    vocab: 字典

    返回: (mlm_input_tokens, pred_positions_and_labels)
    mlm_input_tokens: 处理后的tokens, 15%的tokens已经做了替换
                      e.g. ['<cls>', 'this', 'movie', '<mask>', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    pred_positions_and_labels: list of (mlm_pred_position, token)
                               mlm_pred_position: 需要预测的位置, e.g. 3
                               token: 需要预测的标签, e.g. 'is'
    """
    # 为遮蔽语言模型的输入创建新的token副本，其中输入可能包含替换的'<mask>'或随机token
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []  # [(mlm_pred_position, token)]
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机token进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            # 已经预测足够的tokens, 返回
            break
        masked_token = None
        # 80%的时间: 将token替换为<mask>
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间: 保token不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间: 用随机token替换该token
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token  # 替换成masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    """
    处理一个tokens, 返回训练MLM的数据

    参数:
    tokens: e.g. ['<cls>', 'this', 'movie', 'is', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    vocab: 字典

    返回: (mlm_input_tokens_ids, pred_positions, mlm_pred_labels_ids)
    mlm_input_tokens_ids: 输入tokens的索引
                          e.g. vocab[['<cls>', 'this', 'movie', '<mask>', 'great', '<sep>', 'i', 'like', 'it', '<sep>']]
    pred_positions: 需要预测的位置索引, e.g. [3, ...]
    mlm_pred_labels_ids: 预测的标签索引, e.g. vocab[['is', ...]]
    """
    candidate_pred_positions = []  # list of int
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊token
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语言模型任务中预测15%的随机token
    num_mlm_preds = max(1, round(len(tokens) * 0.15))

    # mlm_input_tokens: 处理后的tokens, 15%的tokens已经做了替换
    #                   e.g. ['<cls>', 'this', 'movie', '<mask>', 'great', '<sep>', 'i', 'like', 'it', '<sep>']
    # pred_positions_and_labels: list of (mlm_pred_position, token)
    #                            mlm_pred_position: 需要预测的位置, e.g. 3
    #                            token: 需要预测的标签, e.g. 'is'
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels
                      ]  # list of int, e.g. [3, ...]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels
                       ]  # list of token, e.g. ['is', ...]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    """
    填充样本

    参数:
    examples: [(mlm_input_tokens_ids, pred_positions, mlm_pred_labels_ids, segments, is_next)]
              mlm_input_tokens_ids: e.g. vocab[['<cls>', 'this', 'movie', '<mask>', 'great', '<sep>', 'i', 'like', 'it', '<sep>']]
              pred_positions: e.g. [3, ...]
              mlm_pred_labels_ids: e.g. vocab[['is', ...]]
              segments: e.g. [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
              is_next: True | False
    max_len: 最大长度
    vocab: 字典

    返回: (all_token_ids, all_segments, valid_lens, all_pred_positions,
           all_mlm_weights, all_mlm_labels, nsp_labels)
    all_token_ids的形状: (num_examples, max_len), 每个token_ids长度为max_len, 长度不足的用<pad>补足
    all_segments的形状: (num_examples, max_len), 每个segments的长度为max_len, 长度不足的用0补足
    valid_lens的形状: (num_examples, ), 每个token_ids的有效长度, 不包括<pad>
    all_pred_positions: (num_examples, max_num_mlm_preds), 每个pred_positions长度为max_num_mlm_preds, 长度不足的用0补足
    all_mlm_weights: (num_examples, max_num_mlm_preds), 有效的pred_positions对应的权重为1, 填充对应的权重为0
    all_mlm_labels: (num_examples, max_num_mlm_preds), 每个pred_label_ids长度为max_num_mlm_preds, 长度不足的用0补足
    nsp_labels: (num_examples, )
    """
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens, = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(
            torch.tensor(token_ids + [vocab['<pad>']] *
                         (max_len - len(token_ids)),
                         dtype=torch.long))
        all_segments.append(
            torch.tensor(segments + [0] * (max_len - len(segments)),
                         dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(
            torch.tensor(pred_positions + [0] *
                         (max_num_mlm_preds - len(pred_positions)),
                         dtype=torch.long))
        # 填充token的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] *
                         (max_num_mlm_preds - len(pred_positions)),
                         dtype=torch.float32))
        all_mlm_labels.append(
            torch.tensor(mlm_pred_label_ids + [0] *
                         (max_num_mlm_preds - len(mlm_pred_label_ids)),
                         dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)


class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        """
        参数:
        paragraphs: 段落的列表, 每个元素是多个句子列表, e.g. ['this moive is great', 'i like it']
        max_len: 最大长度
        """
        # 处理前的paragraphs[i]表示句子的列表, e.g. ['this moive is great', 'i like it']
        # 经过处理后的paragraphs[i]表示一个段落句子的token列表, e.g. [['this', 'movie', 'is', 'great'], ['i', 'like', 'it']]
        paragraphs = [
            tokenize(paragraph, token='word') for paragraph in paragraphs
        ]
        # 经过处理后的sentences[i]表示一个句子的token列表, e.g. ['this', 'movie', 'is', 'great']
        sentences = [
            sentence for paragraph in paragraphs for sentence in paragraph
        ]

        # 构建Vocab
        self.vocab = Vocab(
            sentences,
            min_freq=5,
            reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])

        examples = []  # 训练样本的集合
        # 获取下一句子预测任务的数据
        # 此时的examples: [(tokens, segments, is_next)]
        for paragraph in paragraphs:
            examples.extend(
                _get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab,
                                             max_len))

        # 获取遮蔽语言模型任务的数据
        # 此时的examples: [(mlm_input_tokens_ids, pred_positions, mlm_pred_labels_ids, segments, is_next)]
        # mlm_input_tokens_ids: e.g. vocab[['<cls>', 'this', 'movie', '<mask>', 'great', '<sep>', 'i', 'like', 'it', '<sep>']]
        # pred_positions: e.g. [3, ...]
        # mlm_pred_labels_ids: e.g. vocab[['is', ...]]
        # segments: e.g. [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        # is_next: True | False
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) +
                     (segments, is_next))
                    for tokens, segments, is_next in examples]

        # 填充输入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights, self.all_mlm_labels,
         self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def load_data_wiki(batch_size, max_len):
    """
    加载WikiText_2数据集
    """
    paragraphs = _read_wiki(ml.download_extract('wikitext_2'))
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set,
                                             batch_size,
                                             shuffle=True)
    return train_iter, train_set.vocab


# 生成模型:
# batch_size, max_len = 512, 64
# train_iter, vocab = load_data_wiki(batch_size, max_len)
# net = BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
#                     ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
#                     num_layers=2, dropout=0.2, key_size=128, query_size=128,
#                     value_size=128, hid_in_features=128, mlm_in_features=128,
#                     nsp_in_features=128)
# devices = ml.try_all_gpus()
# loss = nn.CrossEntropyLoss()


def _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X,
                         valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y,
                         nsp_y):
    """
    针对一个批量的数据:
    1. BertModel前向传播
    2. 计算MLM Loss
    3. 计算NSP Loss
    4. 计算最终Loss = MLM Loss + NPS Loss

    参数:
    net: BERTModel实例
    loss: nn.CrossEntropyLoss实例
    vocab_size: 字典大小
    tokens_X: (batch_size, max_len)
    segments_X: (batch_size, max_len)
    valid_lens_x: (batch_size, )
    pred_positions_X: (batch_size, round(max_len * 0.15))
    mlm_weights_X: (batch_size, round(max_len * 0.15))
    mlm_Y: (batch_size, round(max_len * 0.15))
    nsp_y: (batch_size, )
    """
    # 前向传播
    # mlm_Y_hat的形状: (batch_size, round(max_len * 0.15), vocab_size)
    # nsp_Y_hat的形状: (batch_size, 2)
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1), pred_positions_X)
    # 计算遮蔽语言模型损失
    # 将mlm_Y_hat处理成: (batch_size*round(max_len * 0.15), vocab_size)
    # 将mlm_Y处理成: (batch_size*round(max_len * 0.15),)
    # 将mlm_weights_X处理成: (batch_size*round(max_len * 0.15),)
    # mlm_l的形状: (batch_size*round(max_len * 0.15),)
    loss.reduction = 'none'
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size),
                 mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1, )
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)  # 自己计算mean

    # 计算下一句子预测任务的损失
    loss.reduction = 'mean'
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l  # MLM Loss和NSP Loss可以分别乘以weight再相加, 我们实现的版本直接相加了
    print('loss:', mlm_l, nsp_l)
    return mlm_l, nsp_l, l


def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    """
    训练BERT

    参数:
    train_iter: 训练数据迭代器
    net: BERTModel实例
    loss: nn.CrossEntropyLoss实例
    vocab_size: 字典大小
    devices: todo
    num_steps: 训练多少个batch
    """
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, ml.Timer()
    animator = ml.Animator(xlabel='step',
                           ylabel='loss',
                           xlim=[1, num_steps],
                           legend=['mlm', 'nsp'])
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = ml.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(net, loss, vocab_size,
                                                   tokens_X, segments_X,
                                                   valid_lens_x,
                                                   pred_positions_X,
                                                   mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')


# 训练BERT
# train_bert(train_iter, net, loss, len(vocab), devices, 1000)

# 用BERT表示文本
# batch_size, max_len = 512, 64
# train_iter, vocab = load_data_wiki(batch_size, max_len)
# devices = ml.try_all_gpus()
# def get_bert_encoding(net, tokens_a, tokens_b=None):
#     """
#     返回tokens_a和tokens_b中所有token的BERT表示

#     参数:
#     net: BERTModel实例
#     tokens_a: e.g. ['this', 'movie', 'is', 'great']
#     tokens_b: e.g. ['i', 'like', 'it']

#     返回:
#     encoded_X的形状: (1, num_steps, num_hiddens)
#     """
#     tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
#     # token_ids的形状: (1, num_steps)
#     token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
#     # segments的形状: (1, num_steps)
#     segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
#     # valid_len的形状: (1,)
#     valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
#     # encoded_X的形状: (1, num_steps, num_hiddens)
#     encoded_X, _, _ = net(token_ids, segments, valid_len)
#     return encoded_X
