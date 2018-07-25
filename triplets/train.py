import os
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from triplets.data_provider import DataProvider


def main(args):

    # data

    data_provider = DataProvider(args.annotations_csv_path,
                                 args.batch_size,
                                 args.image_embeddings_dir,
                                 args.objects_embeddings_path,
                                 args.vocab_path)
    object_embedding_size = data_provider.get_object_embedding_size()
    num_classes = data_provider.get_num_classes()

    # placeholders

    image_embeddings = tf.placeholder(dtype=tf.float32,
                                      shape=(None, args.image_embedding_size),
                                      name='image_embeddings')
    object_embeddings = tf.placeholder(dtype=tf.float32,
                                       shape=(None, object_embedding_size * 2),
                                       name='object_embeddings')
    relationships_labels = tf.placeholder(dtype=tf.int32,
                                          shape=(None,),
                                          name='relationships_labels')

    # Build model

    images_reduction = slim.fully_connected(inputs=image_embeddings, num_outputs=512)
    objects_reduction_1 = slim.fully_connected(inputs=object_embeddings[:,0:object_embedding_size], num_outputs=512)
    objects_reduction_2 = slim.fully_connected(inputs=object_embeddings[:, object_embedding_size:], num_outputs=512)

    embeddings_concat = tf.concat([image_embeddings, objects_reduction_1, objects_reduction_2], axis=1)

    net = embeddings_concat

    with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=args.weight_decay)):

        net = slim.fully_connected(inputs=net, num_outputs=1024)
        net = slim.fully_connected(inputs=net, num_outputs=1024)
        net = slim.fully_connected(inputs=net, num_outputs=num_classes)

    # loss

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=relationships_labels, logits=net)
    reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    tf.summary.scalar('softmax_loss', loss)
    tf.summary.scalar('reg_loss', reg_loss)

    loss = loss + reg_loss

    # accuracy

    pass

    # train op

    train_op = tf.train.AdamOptimizer().minimize(loss)

    # train

    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        summary_writer = tf.summary.FileWriter(args.logs_dir, sess.graph)

        for epoch in range(50):
            data_provider.next_epoch()

            for batch in range(0, 1000):
                batch_image_embeddings, batch_object_embeddings, batch_labels = data_provider.next_batch()
                feed_dict = {
                    images_reduction: batch_image_embeddings,
                    object_embeddings: batch_object_embeddings,
                    relationships_labels: batch_labels
                }

                _, loss = sess.run([train_op, loss], feed_dict=feed_dict)

                print('Loss: {}'.format(loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--annotations_csv_path', type=str, required=True)
    parser.add_argument('--image_embeddings_dir', type=str, required=True)
    parser.add_argument('--objects_embeddings_path', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--logs_dir', type=str, required=True)

    parser.add_argument('--image_embedding_size', type=int, default=4096)

    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=16)

    main(parser.parse_args())