import time
import tensorflow as tf

import utils.config
from seq2seq import seq2seq
from utils.utils import load_model, config_gpu


def train_model(model, params, ckpt):
    _, embedding_matrix, vocab_i2w, vocab_w2i = load_model(params['vector_train_method'])

    batch_size = params['batch_size']
    pad_index = vocab_w2i['<pad>']
    unk_index = vocab_w2i['<unk>']

    # 优化器
    optimizer = tf.keras.optimizers.Adagrad(params['learning_rate'],
                                            initial_accumulator_value=params['adagrad_init_acc'],
                                            clipnorm=params['max_grad_norm'])

    # 交叉熵计算
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred):
        pad_mask = tf.math.equal(real, pad_index)
        nuk_mask = tf.math.equal(real, unk_index)
        # 计算真实logit位置掩码
        mask = tf.math.logical_not(tf.math.logical_or(pad_mask, nuk_mask))
        # 计算loss向量
        loss_ = loss_object(real, pred)
        # mask适配loss的数据类型，两数相乘完成掩码操作
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        # 返回平均loss值
        return tf.reduce_mean(loss_)

    @tf.function
    def batch_train(enc_input, report):
        """
        批训练
        :param enc_input: 一个batch的x
        :param report: 一个batch的y
        """
        with tf.GradientTape() as gt:
            enc_output, enc_hidden = model.call_encoder(enc_input)  # 进行编码
            dec_hidden = enc_hidden  # 第一个解码器隐层输入
            dec_input = tf.expand_dims([vocab_w2i['<start>']] * batch_size, 1)
            dec_output, dec_hidden = model.call_decoder(dec_input, dec_hidden, enc_output, report)
            loss = loss_function(report[:, 1:], dec_output)
            # 反向梯度求导
            variables = model.encoder.trainable_variables + model.decoder.trainable_variables + model.attention.trainable_variables
            gradients = gt.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            return loss

    dataset, steps_per_epoch = train_batch_generator(batch_size)
    for epoch in range(params['train_epochs']):
        print('阶段 {} 开始训练'.format(epoch + 1))
        start = time.time()
        total_loss = 0
        for (batch, (batch_x, batch_y)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = batch_train(batch_x, batch_y)
            total_loss += batch_loss
            if batch % 100 == 0:
                print('阶段 {} 第 {} 批数据完成， 总Loss {:.4f}'.format(epoch + 1, batch + 1, batch_loss))
        print('阶段 {} / {} 完成, 总Loss {:.4f}'.format(epoch + 1, params['train_epochs'], total_loss / steps_per_epoch))
        print('此阶段耗时 {} 秒\n'.format(time.time() - start))
        ckpt.save()
        print('纪录checkpoint...\n')


def train(params):
    config_gpu()  # 配置GPU环境

    # 构建模型
    print("创建模型 ...")
    model = seq2seq(params=params)

    # 获取保存管理者
    print("创建模型保存器")
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, config.checkpoint_path, max_to_keep=3)
    if checkpoint_manager.latest_checkpoint:
        print("加载最新保存器数据 {} ...".format(checkpoint_manager.latest_checkpoint))
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
    else:
        print("初始化保存器.")
    # 训练模型
    print("开始训练 ...")
    train_model(model, params, checkpoint_manager)


if __name__ == '__main__':
    params = config.get_params()
    params['batch_size'] = 512
    params['train_epochs'] = 15
    params['learning_rate'] = 0.01
    train(params)