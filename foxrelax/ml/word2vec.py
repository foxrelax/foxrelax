import os
import collections
import math
import random
from IPython import display
import torch
from torch import nn
from torch.nn import functional as F
from foxrelax.ml import torch as ml

display.set_matplotlib_formats('svg')


def read_ptb():
    """
    将PTB数据集加载到文本行的列表中(这个数据集一共42069句话)

    ptb.zip
    ptb.test.txt - 440K
    ptb.train.txt - 4.9M
    ptb.valid.txt - 391K

    Examples:
    >>> sentences = read_ptb()
    >>> len(sentences) # 显示一共多少句话
    42069

    >>> sentences[:3] # 显示前三句话
    [['aer', 'banknote', 'berlitz', 'calloway', 'centrust', 'cluett',
      'fromstein', 'gitano', 'guterman', 'hydro-quebec', 'ipo', 'kia',
      'memotec', 'mlx', 'nahb', 'punts', 'rake', 'regatta', 'rubens',
      'sim', 'snack-food', 'ssangyong', 'swapo', 'wachter'],
    ['pierre', '<unk>', 'N', 'years', 'old', 'will', 'join', 'the', 'board',
      'as', 'a', 'nonexecutive', 'director', 'nov.', 'N'],
    ['mr.', '<unk>', 'is', 'chairman', 'of', '<unk>', 'n.v.', 'the', 
      'dutch', 'publishing', 'group']]

    返回:
    outputs: list of sentence
             其中每个sentence是一个token list, 
             e.g. ['mr.', '<unk>', 'is', 'chairman', 'of', '<unk>', 'n.v.', 'the', 
                   'dutch', 'publishing', 'group']
    """
    data_dir = ml.download_extract('ptb')
    # Read the training set.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]


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


# 构建vocab
# sentences = read_ptb()
# vocab = Vocab(sentences, min_freq=10)


def subsample(sentences, vocab):
    """
    下采样高频词, 返回下采样之后的sentences和counter

    1. 过滤掉未知token'<unk>'
    2. 下采样sentences(一定概率的删除高频词, 频率越高, 删除的概率越大)

    参数:
    sentences: list of sentence
               其中每个sentence是一个token list, 
               e.g. ['mr.', '<unk>', 'is', 'chairman', 'of', '<unk>', 'n.v.', 'the', 
                      utch', 'publishing', 'group']
    vocab: 词典

    返回:
    output: (subsampled, counter)
    subsampled: list of sentence, 下采样之后的sentence列表
    counter: collections.Counter实例
    """
    # 过滤掉未知token'<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())

    # 如果在下采样期间保留token, 则返回True
    # 一定概率的删除高频词, 频率越高, 删除的概率越大
    def keep(token):
        return (random.uniform(0, 1) < math.sqrt(
            1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)]
             for line in sentences], counter)


def get_centers_and_contexts(corpus, max_window_size):
    """
    返回Skip-Gram中的中心词和上下文词(针对每一行, 我们使用了一个随机的窗口大小)

    1. 遍历corpus的每一行
    2. 遍历每一行的每一个token, 随机一个window_size进行采样

    Example:
    >>> tiny_dataset = [list(range(7)), list(range(7, 10))]
    dataset [[0, 1, 2, 3, 4, 5, 6], 
            [7, 8, 9]]
    >>> for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    >>>     print('center', center, 'has contexts', context)
    center 0 has contexts [1]
    center 1 has contexts [0, 2, 3]
    center 2 has contexts [0, 1, 3, 4]
    center 3 has contexts [2, 4]
    center 4 has contexts [2, 3, 5, 6]
    center 5 has contexts [3, 4, 6]
    center 6 has contexts [5]
    center 7 has contexts [8, 9]
    center 8 has contexts [7, 9]
    center 9 has contexts [7, 8]

    参数:
    corpus: list of sentence
            其中每个sentence是一个token_id list,
            e.g. [6697, 4127, 993, 1325, 2641, 2340, 4465, 3927, 1773, 1291]
    max_window_size: 采样的滑动窗口大小, 对于每一行数据, 窗口大小是随机的, 范围在: window_size = [1-max_window_size]
                     从中心词向前和向后看最多window_size个单词

    返回: (centers, contexts)
    centers: list of token_id
    contexts: list of context
              其中每个context表示一个中心词对应的上下文词token_id list
    """
    centers, contexts = [], []
    for line in corpus:
        # 要形成"中心词-上下文词"对, 每个句子至少需要有2个词
        if len(line) < 2:
            continue
        centers += line  # 一次性增加了了n个centers
        for i in range(len(line)):  # 上下文窗口中间`i`
            window_size = random.randint(1, max_window_size)  # 随机一个窗口大小
            indices = list(
                range(max(0, i - window_size),
                      min(len(line), i + 1 + window_size)))
            # 从上下文词中排除中心词
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts


class RandomGenerator:
    """
    根据采样权重在{1, 2, ..., len(sampling_weights)}这些token_id中随机采样

    索引为1、2、...（索引0是vocab中排除的未知标记<unk>）
    """
    def __init__(self, sampling_weights):
        """
        参数:
        sampling_weights: list of weight, 对应{1, ..., len(sampling_weights)}这些
                          token_id的采样权重
        """
        # 索引为1、2、...（索引0是vocab中排除的未知标记<unk>）
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights  # 采样权重
        self.candidates = []  # list of token_id
        self.i = 0

    def draw(self):
        """
        采样(返回一个随机的token_id)

        在实现的时候, 会一次性采样出k=10000个token_id缓存起来, 返回结果的时候
        从缓存中直接返回就行, 当10000都用过一边之后, 再重新采样

        返回:
        output: token_id
        """
        if self.i == len(self.candidates):
            # 根据sampling_weights来采样:
            # 一次性采样出来, 缓存`k`个随机采样结果
            self.candidates = random.choices(self.population,
                                             self.sampling_weights,
                                             k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


def get_negatives(all_contexts, vocab, counter, K):
    """
    返回负采样中的噪声词

    参数:
    all_contexts: list of context
                  其中每个context表示一个中心词对应的上下文词token_id list
    vocab: 字典
    counter: collections.Counter实例
    K: 负采样的参数, 也就是一个正样本对应多少个负样本(通常为5)

    返回:
    all_negatives: list of negative
                   其中每个negative表示一个中心词对应的负样本token_id list
                   len(all_contexts) == len(all_negatives)
    """

    # 为每个token_id{1、2、..., len(vocab)-1}（索引0是vocab中排除的未知标记<unk>）
    # 生成对应的采样权重
    sampling_weights = [
        counter[vocab.to_tokens(i)]**0.75 for i in range(1, len(vocab))
    ]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []  # 负样本列表
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 注意: 我们做了特殊处理, 噪声词不能是上下文词
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


def batchify(data):
    """
    返回带有负采样的Skip-Gram的小批量样本(作为torch.utils.data.DataLoader.collate_fn使用)

    会自动计算max_len, max_len为这个批量中: max(len(context) + len(negative))

    输入:
    data: 一个批量的(center, context, negative)
           center: token_id
           context: 表示一个中心词对应的上下文词token_id list
           negative: 表示一个中心词对应的负样本token_id list
    
    返回:
    output: (centers, contexts_negatives, masks, labels)
            centers的形状: (batch_size, 1)
            contexts_negatives的形状: (batch_size, max_len) 包含了正样本 + 负样本, 长度不足的补0
            masks的形状: (batch_size, max_len), 长度为max_len
            labels的形状: (batch_size, ), 标签, 长度为max_len, 长度不足的补0
    """
    # 计算最大的长度
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    # 返回: (centers, contexts_negatives, masks, labels)
    return (torch.tensor(centers).reshape(
        (-1, 1)), torch.tensor(contexts_negatives), torch.tensor(masks),
            torch.tensor(labels))


def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """
    下载PTB数据集，然后将其加载到内存中

    >>> names = ['centers', 'contexts_negatives', 'masks', 'labels']
    >>> data_iter, vocab = load_data_ptb(512, 5, 5)
    >>> for batch in data_iter:
    >>>     for name, data in zip(names, batch):
    >>>         print(name, 'shape:', data.shape)
    >>>     break
    centers shape: torch.Size([512, 1])
    contexts_negatives shape: torch.Size([512, 60])
    masks shape: torch.Size([512, 60])
    labels shape: torch.Size([512, 60])
    """
    sentences = read_ptb()
    vocab = Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(all_contexts, vocab, counter,
                                  num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset,
                                            batch_size,
                                            shuffle=True,
                                            collate_fn=batchify)
    return data_iter, vocab


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    """
    Skip-Gram的前向传播过程, 也就是center对应的词向量和contexts_and_negatives所有的
    正样本和负样本对应的词向量做点积
    
    参数:
    center的形状: (batch_size, 1) 中心词索引
    contexts_and_negatives的形状: (batch_size, max_len) 上下文与噪声词索引
    embed_v: 嵌入层, (vocab_size, embed_size)
    embed_u: 嵌入层, (vocab_size, embed_size)

    返回:
    pred的形状: (batch_size, 1, max_len)
    """

    # v的形状: (batch_size, 1, embed_size) - 词向量
    # u的形状: (batch_size, max_len, embed_size) 词向量
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)

    # 做点积
    # pred的形状: (batch_size, 1, max_len)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


class SigmoidBCELoss(nn.Module):
    """
    带掩码的二元交叉熵损失
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        """
        参数:
        inputs的形状: (batch_size, max_len)
        target的形状: (batch_size, max_len)
        mask的形状: (batch_size, max_len)

        返回:
        out的形状: (batch_size, )
        """
        out = nn.functional.binary_cross_entropy_with_logits(inputs,
                                                             target,
                                                             weight=mask,
                                                             reduction='none')
        return out.mean(dim=1)


# 构建网络:
# embed_size = 100
# net = nn.Sequential(
#     nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size),
#     nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size))

loss = SigmoidBCELoss()


def train(net, data_iter, lr, num_epochs, device=ml.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = ml.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs])
    # 归一化的损失之和, 归一化的损失数
    metric = ml.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = ml.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            # center的形状: (batch_size, 1)
            # context_negative的形状: (batch_size, max_len) 包含了正样本 + 负样本, 长度不足的补0
            # mask的形状: (batch_size, max_len), 长度为max_len
            # label的形状: (batch_size, ), 标签, 长度为max_len, 长度不足的补0
            center, context_negative, mask, label = [
                data.to(device) for data in batch
            ]

            # pred的形状: (batch_size, 1, max_len)
            pred = skip_gram(center, context_negative, net[0], net[1])
            # 1. 将pred的形状转换成: (batch_size, max_len)在送入loss
            # 2. mask.shape[1] - 表示一行一共多少元素
            #    mask.sum(axis=1) - 表示一行为1的元素有多少个
            #    通过这种方式, 我们计算出真正的mean loss, 不受有效样本个数的影响
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask) /
                 mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], ))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')


# lr, num_epochs = 0.002, 5
# train(net, data_iter, lr, num_epochs)


def get_similar_tokens(query_token, k, embed, vocab):
    """
    参数:
    query_token: str, 需要查询的token
    k: 需要查询最近的k个词
    embed: embed_u
    vocab: 词典

    返回:
    """
    # W的形状: (vocab_size, embed_size)也就是(vocab_size, 100), 表示vocab中所有的词的词向量
    # x的形状: (embed_size,)也就是(100,)
    W = embed.weight.data
    x = W[vocab[query_token]]

    # 一次性计算出x和vocab中所有的词(W)的余弦相似性. 增加1e-9以获得数值稳定性
    # cos的形状: (vocab_size,)
    cos = torch.mv(
        W, x) / torch.sqrt(torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9)
    # 选出top k+1个最接近的
    topk = torch.topk(cos, k=k + 1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 删除输入词(因为自己和自己是最接近的)
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')


# 测试:
# get_similar_tokens('chip', 3, net[0], vocab)
# cosine sim=0.673: desktop
# cosine sim=0.652: intel
# cosine sim=0.627: microprocessor
