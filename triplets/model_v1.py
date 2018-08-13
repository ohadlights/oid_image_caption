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
