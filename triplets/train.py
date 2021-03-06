import os
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
from triplets.data_provider import DataProvider
from triplets.model_v1 import build_model
from triplets.evaluate_triplets_model import evaluate_model


def print_train_status(sess,
                       batch,
                       batch_count,
                       loss_value,
                       accuracy_value,
                       summary,
                       feed_dict,
                       summary_writer,
                       iteration,
                       log_events_each_iterations=500):

    if batch % log_events_each_iterations == 0:
        # Print status to stdout.
        print('[{0}/{1}] loss = {2:.3f} / accuracy = {3:.3f}'.format(batch, batch_count, loss_value, accuracy_value))

        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=iteration)
        summary_writer.flush()


def main(args):

    # data

    data_provider = DataProvider(args.annotations_csv_path,
                                 args.batch_size,
                                 args.image_embeddings_dir,
                                 args.objects_embeddings_path,
                                 args.vocab_path)
    object_embedding_size = data_provider.get_object_embedding_size()
    num_classes = data_provider.get_num_classes()

    val_data_provider = DataProvider(args.annotations_csv_path_val,
                                     args.batch_size,
                                     args.image_embeddings_dir,
                                     args.objects_embeddings_path,
                                     args.vocab_path)

    # placeholders

    global_step = tf.Variable(0, name='global_step', trainable=False)

    image_embeddings = tf.placeholder(dtype=tf.float32,
                                      shape=(None, args.image_embedding_size),
                                      name='image_embeddings')
    object_embeddings = tf.placeholder(dtype=tf.float32,
                                       shape=(None, 2, object_embedding_size),
                                       name='object_embeddings')
    relationships_labels = tf.placeholder(dtype=tf.int32,
                                          shape=(None,),
                                          name='relationships_labels')

    # Build model

    net = build_model(image_embeddings=image_embeddings,
                      object_embeddings=object_embeddings,
                      num_classes=num_classes,
                      weight_decay=args.weight_decay,
                      is_training=True)

    # loss

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=relationships_labels, logits=net))
    reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    tf.summary.scalar('loss/softmax_loss', loss)
    tf.summary.scalar('loss/reg_loss', reg_loss)

    loss = loss + reg_loss

    # accuracy

    predictions = slim.softmax(net)
    equality = tf.equal(tf.argmax(predictions, 1, output_type=tf.int32), relationships_labels)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    tf.summary.scalar('accuracy/train', accuracy)

    # train op

    train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss, global_step=global_step)

    # summary

    summary = tf.summary.merge_all()

    # Create a saver for writing training checkpoints.

    saver = tf.train.Saver(max_to_keep=5)

    # train

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        latest = tf.train.latest_checkpoint(args.logs_dir)
        if latest:
            print('restore checkpoint: {}'.format(latest))
            saver.restore(sess, latest)

        summary_writer = tf.summary.FileWriter(args.logs_dir, sess.graph)

        for epoch in range(args.epoches):
            data_provider.next_epoch()

            for batch in range(0, args.steps_per_epoch):
                batch_image_embeddings, batch_object_embeddings, batch_labels = data_provider.next_batch()
                feed_dict = {
                    image_embeddings: batch_image_embeddings,
                    object_embeddings: batch_object_embeddings,
                    relationships_labels: batch_labels
                }

                _, loss_value, accuracy_value = sess.run([train_op, loss, accuracy], feed_dict=feed_dict)

                iteration = tf.train.global_step(sess, global_step)

                print_train_status(sess, batch, args.steps_per_epoch, loss_value, accuracy_value, summary, feed_dict,
                                   summary_writer, iteration, 250)

            saver.save(sess=sess, save_path=os.path.join(args.logs_dir, 'model.ckpt'), global_step=global_step)

            # evaluate

            # evaluate_model(predictions,
            #                image_embeddings=image_embeddings,
            #                object_embeddings=object_embeddings,
            #                data_provider=val_data_provider,
            #                sess=sess,
            #                batch_size=args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--annotations_csv_path', type=str, required=True)
    parser.add_argument('--annotations_csv_path_val', type=str, required=True)
    parser.add_argument('--image_embeddings_dir', type=str, required=True)
    parser.add_argument('--objects_embeddings_path', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--logs_dir', type=str, required=True)

    parser.add_argument('--image_embedding_size', type=int, default=4320)

    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoches', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=10000)

    main(parser.parse_args())
