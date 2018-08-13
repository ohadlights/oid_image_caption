from tqdm import tqdm
import tensorflow as tf
import numpy as np


def evaluate_model(predictions,
                   image_embeddings,
                   object_embeddings,
                   data_provider,
                   sess,
                   batch_size,
                   max_samples=100000):

    # call next_epoch to shuffle the data

    data_provider.next_epoch()

    # predict on examples

    all_labels = np.zeros((data_provider.get_num_examples()))
    all_preds = np.zeros((data_provider.get_num_examples(), data_provider.get_num_classes()))

    num_examples = min(max_samples, data_provider.get_num_examples())
    for batch in tqdm(range(0, num_examples, batch_size)):
        batch_image_embeddings, batch_object_embeddings, batch_labels = data_provider.next_batch()

        all_labels[batch:batch + batch_size] = batch_labels

        feed_dict = {
            image_embeddings: batch_image_embeddings,
            object_embeddings: batch_object_embeddings,
        }

        results = sess.run(predictions, feed_dict=feed_dict)
        all_preds[batch:batch + batch_size, :] = results

    # accuracy

    equality = tf.equal(tf.argmax(all_preds, 1, output_type=tf.int32), all_labels)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    accuracy = sess.run(accuracy)
    print('accuracy = {}'.format(accuracy))
    tf.summary.scalar('accuracy/validation', accuracy)
