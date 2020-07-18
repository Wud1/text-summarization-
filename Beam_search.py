import tensorflow as tf
import pandas as pd
import time
from tqdm import tqdm


from seq2seq import seq2seq
from utils.utils import  load_data, config_gpu


def merge_batch_beam(t: tf.Tensor):
    # 输入: shape [batch_size, beam_size ...]
    # 输出: shape [batch_size * beam_size, ...]
    batch_size, beam_size = t.shape[0], t.shape[1]
    return tf.reshape(t, shape=[batch_size * beam_size] + list(t.shape[2:]))


def split_batch_beam(t: tf.Tensor, beam_size: int):
    # 输入: shape [batch_size * beam_size ...]
    # 输出: shape [batch_size, beam_size, ...]
    return tf.reshape(t, shape=[-1, beam_size] + list(t.shape[1:]))


def tile_beam(t: tf.Tensor, beam_size: int):
    # 输入: shape [batch_size, ...]
    # 输出: shape [batch_size, beam_size, ...]
    multipliers = [1, beam_size] + [1] * (t.shape.ndims - 1)
    return tf.tile(tf.expand_dims(t, axis=1), multipliers)


class Hypothesis(object):
    def __init__(self, tokens, log_probs, hidden):
        # 记录所有时间步长0-t
        self.tokens = tokens
        self.log_probs = log_probs
        self.hidden = hidden

    def extend(self, token, log_prob, hidden):
        tokens = self.tokens + [token]
        log_probs = self.log_probs + [log_prob]
        return Hypothesis(tokens, log_probs, hidden)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def tot_log_probs(self):
        return sum(self.log_probs)

    @property
    def avg_log_probs(self):
        return self.tot_log_probs / len(self.log_probs)


def beam_decode(model, params, inp):
    _, _, vocab_i2w, vocab_w2i = load_model(params['vector_train_method'])

    def decode_one_topk(dec_inp, dec_hid, enc_out, k: int = 1):
        # 单步解码
        pred, dec_hid, context_vector, attention_weights = model.call_one_step_decoder(dec_inp, dec_hid, enc_out)
        # 计算top-K概率 - logits和对应的index
        topk_probs, topk_ids = tf.nn.top_k(pred, k, sorted=True)
        # 计算log概率 - logits
        topk_log_probs = tf.math.log(topk_probs)
        # 返回结果
        return topk_log_probs, topk_ids, dec_hid

    # 初始化mask
    start_index = vocab_w2i['<start>']
    stop_index = vocab_w2i['<end>']
    batch_size = len(inp)

    enc_input = tf.convert_to_tensor(inp)
    dec_input = tf.expand_dims([start_index] * batch_size, 1)
    min_steps = params['min_y_length']
    max_steps = params['max_y_length']
    beam_size = params['beam_size']
    end_token = stop_index

    # 编码器输出
    # enc_output: [batch_size, sequence_length, enc_units]
    # enc_hidden: [batch_size, enc_units]
    enc_output, enc_hidden = model.call_encoder(enc_input)

    # 将编码器输出复制beam_size份
    # 并调整维度为[beam_size*batch_size, ...]
    enc_output = merge_batch_beam(tile_beam(enc_output, beam_size))

    # 复制隐层状态
    dec_hidden = enc_hidden

    # 初始化[batch_size, beam_size]个Hypothesis对象
    hyps = [[Hypothesis(tokens=list(dec_input[i].numpy()), log_probs=[0.], hidden=dec_hidden[i])
             for _ in range(beam_size)] for i in range(batch_size)]

    # 进行搜索
    for step in range(max_steps):
        # 获得上一步的输出: [batch_size, beam_size]
        latest_tokens = tf.stack(
            [tf.stack([h.latest_token for h in beam], axis=0) for beam in hyps],
            axis=0
        )
        # 构建解码器单步输入: [batch_size*beam_size, 1]
        dec_input = tf.expand_dims(merge_batch_beam(latest_tokens), axis=1)

        # 获得上一步的隐层: [batch_size, beam_size, dec_units]
        hiddens = tf.stack(
            [tf.stack([h.hidden for h in beam], axis=0) for beam in hyps],
            axis=0
        )

        # 构建解码器隐层[batch_size*beam_size, dec_units]
        dec_hidden = merge_batch_beam(hiddens)

        # 单步解码
        top_k_log_probs, top_k_ids, dec_hidden = \
            decode_one_topk(dec_input, dec_hidden, enc_output, k=beam_size)

        # 将上述结果形状变为[batch_size, beam_size, ...]
        top_k_log_probs = split_batch_beam(top_k_log_probs, beam_size)
        top_k_ids = split_batch_beam(top_k_ids, beam_size)
        dec_hidden = split_batch_beam(dec_hidden, beam_size)

        # 遍历batch中所有句子:
        for bc in range(batch_size):
            # 当前句子对应的变量
            bc_hyps = hyps[bc] # bc=1
            bc_top_k_log_probs = top_k_log_probs[bc]
            bc_top_k_ids = top_k_ids[bc]
            bc_dec_hidden = dec_hidden[bc]

            # 遍历上一步中所有的假设情况: beam_size个
            # 获得当前步骤的最大概率假设: beam_size * k个 (k = beam_size)
            bc_all_hyps = []
            num_prev_bc_hyps = 1 if step == 0 else len(bc_hyps)
            for i in range(num_prev_bc_hyps):
                hyp, new_hidden = bc_hyps[i], bc_dec_hidden[i]
                # 分裂，增加当前步中的beam_size * k个可能假设 (k = beam_size)
                for j in range(beam_size):
                    new_hyp = hyp.extend(token=bc_top_k_ids[i, j].numpy(),
                                         log_prob=bc_top_k_log_probs[i, j].numpy(),
                                         hidden=new_hidden)
                    bc_all_hyps.append(new_hyp)

            # 重置当前句子对应的Hypothesis对象列表
            bc_hyps = []

            # 按照概率排序
            sorted_bc_hyps = sorted(bc_all_hyps, key=lambda h: h.avg_log_probs, reverse=True)

            # 筛选top-'beam_size'句话
            for h in sorted_bc_hyps:
                bc_hyps.append(h)
                if len(bc_hyps) == beam_size:
                    # 假设句子数目达到beam_size, 则不再添加
                    break

            # 更新hyps
            hyps[bc] = bc_hyps

    # 从获得的假设集中取出最终结果
    results = [[]] * batch_size

    # 遍历所有句子
    for bc in range(batch_size):
        # 当前句子对应的变量
        bc_hyps = hyps[bc]
        # 优先选取有结束符的结果
        for i in range(beam_size):
            hyp = bc_hyps[i]
            tokens = hyp.tokens[:]
            if end_token in tokens:
                tokens = tokens[1:tokens.index(end_token)]
                # 有结束符且满足最小长度要求
                if len(tokens) > min_steps:
                    results[bc] = tokens
                    break
        # 如果找到了满足要求的结果，则直接处理下一句
        if results[bc]:
            continue
        # 若在上述条件下未找到合适结果，则只找没有结束符的结果
        for i in range(beam_size):
            hyp = bc_hyps[i]
            tokens = hyp.tokens[:]
            if end_token in tokens:
                continue
            results[bc] = tokens[1:]

    def get_abstract(ids):
        return " ".join([vocab_i2w[index] for index in ids])
    # 返回结果
    return [get_abstract(res) for res in results]


def beam_search(params):
    config_gpu()  # 配置GPU
    # 加载数据集、模型
    _, _, test_X = load_data()
    model = Seq2Seq(params)

    start = time.time()
    print('使用集束搜索开始预测...')
    results = []
    dataset, steps_per_epoch = test_batch_generator(params['batch_size'])
    with tqdm(total=steps_per_epoch, position=0, leave=True) as tq:
        for (batch, batch_x) in enumerate(dataset.take(steps_per_epoch)):
            results += beam_decode(model, params, batch_x)
            tq.update(1)

    print('预测完成，耗时{}s\n处理至文件...'.format(time.time() - start))

    def result_proc(text):
        """
        对预测结果做最后处理
        :param text: 单条预测结果
        :return:
        """
        # text = text.lstrip(' ，！。')
        text = text.replace(' ', '')
        text = text.strip()
        if '<end>' in text:
            text = text[:text.index('<end>')]
        return text
    test_csv = pd.read_csv(config.test_set, encoding="UTF-8")
    # 赋值结果
    test_csv['Prediction'] = results
    # 提取ID和预测结果两列
    test_df = test_csv[['QID', 'Prediction']]
    # 结果处理
    test_df['Prediction'] = test_df['Prediction'].apply(result_proc)
    # 保存结果
    test_df.to_csv(config.inference_result_path, index=None, sep=',')
    print('已保存文件至{}'.format(config.inference_result_path))


if __name__ == '__main__':
    params = config.get_params()
    params['batch_size'] = 8
    params['beam_size'] = 8
    beam_search(params)