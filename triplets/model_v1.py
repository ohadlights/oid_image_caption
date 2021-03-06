import tensorflow as tf
import tensorflow.contrib.slim as slim


def build_model(image_embeddings,
                object_embeddings,
                num_classes,
                weight_decay=0.0001,
                is_training=True):

    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'scale': True,
        'is_training': is_training,
    }

    with slim.arg_scope([slim.fully_connected],
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=tf.nn.relu,
                        trainable=is_training,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.dropout],
                            is_training=is_training):

            images_reduction = tf.identity(image_embeddings)
            # images_reduction = slim.dropout(inputs=images_reduction, keep_prob=0.8)
            images_reduction = slim.fully_connected(inputs=images_reduction, num_outputs=1024, scope='images_reduction_0')
            images_reduction = slim.fully_connected(inputs=images_reduction, num_outputs=512, scope='images_reduction_1')

            objects_reduction = tf.identity(object_embeddings)
            # objects_reduction = slim.dropout(inputs=object_embeddings, keep_prob=0.8)
            objects_reduction = slim.fully_connected(objects_reduction, num_outputs=512, scope='objects_reduction_0')
            objects_reduction = slim.fully_connected(objects_reduction, num_outputs=512, scope='objects_reduction_1')
            objects_reduction = slim.flatten(objects_reduction, scope='objects_reduction_flatten')

            embeddings_concat = tf.concat([images_reduction, objects_reduction], axis=1, name='embeddings_concat')

            net = embeddings_concat

            net = slim.fully_connected(inputs=net, num_outputs=512)
            # net = slim.dropout(inputs=net, keep_prob=0.5)
            net = slim.fully_connected(inputs=net, num_outputs=256)
            net = slim.dropout(inputs=net, keep_prob=0.5)
            net = slim.fully_connected(inputs=net, num_outputs=num_classes, activation_fn=None)

            return net


class ModelV1:
    def __init__(self,
                 image_embedding_size,
                 object_embedding_size,
                 num_classes,
                 checkpoints_dir=None):

        self.image_embeddings = tf.placeholder(dtype=tf.float32,
                                               shape=(None, image_embedding_size),
                                               name='image_embeddings')
        self.object_embeddings = tf.placeholder(dtype=tf.float32,
                                                shape=(None, 2, object_embedding_size),
                                                name='object_embeddings')

        self.net = build_model(image_embeddings=self.image_embeddings,
                               object_embeddings=self.object_embeddings,
                               num_classes=num_classes,
                               is_training=False)

        self.predictions = tf.nn.softmax(self.net)

        self.checkpoints_dir = checkpoints_dir

        self.sess = None

    def __enter__(self):
        self.sess = tf.Session()
        if self.checkpoints_dir:
            latest = tf.train.latest_checkpoint(self.checkpoints_dir)
            print('restore checkpoint: {}'.format(latest))
            tf.train.Saver().restore(self.sess, latest)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()
