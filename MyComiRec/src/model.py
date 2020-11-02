import os
import tensorflow as tf
from models import positional_encoding


class Model(object):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="DNN"):
        self.model_flag = flag
        self.reg = False
        self.batch_size = batch_size
        self.n_mid = n_mid
        self.neg_num = 10
        with tf.name_scope('Inputs'):
            """
            self.uid_batch_ph: 每一批的所有用户id, [uid1, uid2, ..., uid_batch]
            self.mid_batch_ph: 从每一个用户的行为中下一个要点击的item_id,
            self.mid_his_batch_ph: 每一批每个用户行为序列item_id, [[seq_len], [seq_len], ..., [seq_len]],
            self.mask: 标记[[seq_len],[seq_len],[seq_len],[seq_len],...,[seq_len] ]
            """
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])

        self.mask_length = tf.cast(tf.reduce_sum(self.mask, -1), dtype=tf.int32)

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, embedding_dim], trainable=True)
            self.mid_embeddings_bias = tf.get_variable("bias_lookup_table", [n_mid], initializer=tf.zeros_initializer(), trainable=False)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            # (b, sql_len, embedding_dim)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)

        # 正样本嵌入向量
        self.item_eb = self.mid_batch_embedded
        # mask后的用户行为embedding序列
        self.item_his_eb = self.mid_his_batch_embedded * tf.reshape(self.mask, (-1, seq_len, 1))

    # user_emb: 每一个最符合相似的兴趣向量
    def build_sampled_softmax_loss(self, item_emb, user_emb):
        """"
        def sampled_softmax_loss_v2(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            remove_accidental_hits=True,
                            seed=None,
                            name="sampled_softmax_loss"):
        """
        self.loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                self.mid_embeddings_var,
                self.mid_embeddings_bias,
                tf.reshape(self.mid_batch_ph, [-1, 1]),
                user_emb,
                self.neg_num * self.batch_size,
                self.n_mid
            )
        )

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    # model.train(sess, list(data_iter) + [lr])
    def train(self, sess, inps):
        """
        data_iter[0]: 每一批的所有用户id, [uid1, uid2, ..., uid_batch], batch_size
        data_iter[1]: 每一个用户的行为下一个要点击的item_id, batch_size
        data_iter[2]: 每一批每个用户行为序列item_id, [[], [], ..., []], 20(maxLen)
        data_iter[3]: 标记 , 20(maxLen)
        """
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.mid_his_batch_ph: inps[2],
            self.mask: inps[3],
            self.lr: inps[4]
        }
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def output_item(self, sess):
        item_embs = sess.run(self.mid_embeddings_var)
        return item_embs

    def output_user(self, sess, inps):
        user_embs = sess.run(self.user_eb, feed_dict={
            self.mid_his_batch_ph: inps[0],
            self.mask: inps[1]
        })
        return user_embs

    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)

class Model_DNN(Model):
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_DNN, self).__init__(n_mid, embedding_dim, hidden_size,
                                           batch_size, seq_len, flag="DNN")

        masks = tf.concat([tf.expand_dims(self.mask, -1) for _ in range(hidden_size)], axis=-1)
        # item_his_eb: (b, sql_len, embedding_dim) , sum pooling
        self.item_his_eb_mean = tf.reduce_sum(self.item_his_eb, 1) / (tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9)
        self.user_eb = tf.layers.dense(self.item_his_eb_mean, hidden_size, activation=None)
        self.build_sampled_softmax_loss(self.item_eb, self.user_eb)

def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape

class Model_ComiRec_SA(Model):
    # Model_ComiRec_SA(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
    def __init__(self, n_mid, embedding_dim, hidden_size, batch_size, num_interest, seq_len=256, add_pos=True):
        super(Model_ComiRec_SA, self).__init__(n_mid, embedding_dim, hidden_size,
                                                   batch_size, seq_len, flag="ComiRec_SA")

        self.dim = embedding_dim
        item_list_emb = tf.reshape(self.item_his_eb, [-1, seq_len, embedding_dim])

        if add_pos:
            self.position_embedding = tf.expand_dims(positional_encoding(embedding_dim, seq_len), axis=0)
            item_list_add_pos = item_list_emb + tf.tile(self.position_embedding, [tf.shape(item_list_emb)[0], 1, 1])
        else:
            item_list_add_pos = item_list_emb

        num_heads = num_interest
        with tf.variable_scope("self_atten", reuse=tf.AUTO_REUSE) as scope:
            # item_list_add_pos： （b, seq_len, embedding_dim)
            # item_hidden: (b, sql_len, hidden_size * 4)
            item_hidden = tf.layers.dense(item_list_add_pos, hidden_size * 4, activation=tf.nn.tanh)
            # item_att_w: (b, sql_len, num_heads)
            item_att_w  = tf.layers.dense(item_hidden, num_heads, activation=None)
            # item_att_w: (b, num_heads, sql_len)
            item_att_w  = tf.transpose(item_att_w, [0, 2, 1])

            # atten_mask: (b, num_heads, sql_len)
            atten_mask = tf.tile(tf.expand_dims(self.mask, axis=1), [1, num_heads, 1])
            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)

            # 对于填充的位置赋值极小值
            item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
            item_att_w = tf.nn.softmax(item_att_w)

            # item_att_w [batch, num_heads, seq_len]
            # item_list_emb [batch, seq_len, embedding_dim]
            # interest_emb (batch, num_heads, embedding_dim)
            interest_emb = tf.matmul(item_att_w, item_list_emb)

        self.user_eb = interest_emb

        # item_list_emb = [-1, seq_len, embedding_dim]
        # atten: (batch, num_heads, dim) * (batch, dim, 1) = (batch, num_heads, 1)
        atten = tf.matmul(self.user_eb, tf.reshape(self.item_eb, [get_shape(item_list_emb)[0], self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [get_shape(item_list_emb)[0], num_heads]), 1))

        # 找出与target item最相似的用户兴趣向量
        readout = tf.gather(tf.reshape(self.user_eb, [-1, self.dim]), tf.argmax(atten, axis=1, output_type=tf.int32) + tf.range(tf.shape(item_list_emb)[0]) * num_heads)

        self.build_sampled_softmax_loss(self.item_eb, readout)