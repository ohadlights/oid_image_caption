"""
For each image in the test set, you must predict a list of boxes describing objects in the image.
Each box is described as:
    <confidence label_object1 x_min1 y_min1 x_max1 y_max1 label_object2 x_min2 y_min2 x_max2 y_max2 relationship_label>.
The length of your PredictionString should always be a multiple of 12.
Every value is space delimited. The file should contain a header and have the following format:

ImageId,PredictionString
fd162df2a4fdb29d,0.037432 /m/03bt1vf 0.549840 0.603769 0.814588 0.999519 /m/01mzp 0.187824 0.454496 0.245905 0.554354 on 0.044382 /m/03bt1vf 0.549840 0.603769 0.814588 0.999519 /m/01mzpv 0.174735 0.468313 0.238807 0.562794 on
...
"""


import os
import argparse
from itertools import permutations

import numpy as np
from tqdm import tqdm

from triplets.submission.utils import parse_prediction_line
from triplets.model_v1 import ModelV1


def filter_relevant_class_ids(class_ids, boxes):
    return [b for b in boxes if b.class_id in class_ids]


def filter_out_duplicate_class_ids(boxes):
    class_ids = set()
    filtered = []
    for b in boxes:
        if b.class_id not in class_ids:
            class_ids.add(b.class_id)
            filtered += [b]
    return filtered


def build_box_permutations(boxes):
    perms = list(permutations(boxes, r=2))

    # need to remove duplicated perms! (0,1) == (1,0)

    filtered = []
    filtered_keys = set()
    for a, b in perms:
        key_1 = '{}.{}'.format(a.class_id, b.class_id)
        key_2 = '{}.{}'.format(b.class_id, a.class_id)
        if key_1 not in filtered_keys and key_2 not in filtered_keys:
            filtered_keys.add(key_1)
            filtered_keys.add(key_2)
            filtered += [(a, b)]

    return filtered


def build_permutations(input_predictions, class_ids):
    all_permutations = []
    for l in input_predictions:
        boxes_data = l[1]
        boxes = parse_prediction_line(boxes_data)
        boxes = filter_relevant_class_ids(class_ids=class_ids, boxes=boxes)
        boxes = filter_out_duplicate_class_ids(boxes)
        if len(boxes) > 1:
            perms = build_box_permutations(boxes)
            all_permutations += [(l[0], perms)]
        else:
            all_permutations += [(l[0], [])]
    return all_permutations


def main(args):

    # Output

    output_path = os.path.join('submission_files', os.path.basename(args.input_predictions))

    # Generate permutations from input predictions

    class_ids = [l.split(',')[0] for l in open(args.class_ids_path).readlines()]
    input_predictions = [l.strip().split(',') for l in open(args.input_predictions).readlines()[1:]]
    images_perms = build_permutations(input_predictions, class_ids=class_ids)

    # Load embeddings data

    object_embeddings = np.load(args.objects_embeddings_path).astype(np.float32)
    content = [l.strip().split(',') for l in open(args.vocab_path).readlines()]
    object_id_to_embeddings_index = {content[i][0]: [int(a) for a in content[i][1].split()] for i in range(len(content))}

    # Load relationships data

    relationship_index_to_id = {l[1]: l[0] for l in [l.strip().split(',') for l in open(args.relationships_ids_path).readlines()]}

    # Build model

    with ModelV1(image_embedding_size=args.image_embedding_size,
                 object_embedding_size=object_embeddings.shape[1],
                 num_classes=len(relationship_index_to_id),
                 checkpoints_dir=args.checkpoints_dir) as model:

        with open(output_path, 'w') as f:

            for image, perms in tqdm(images_perms):

                f.write('{},'.format(image))

                for box_1, box_2 in perms:

                    image_embedding_path = os.path.join(args.image_embeddings_dir, image + '.jpg.npy')
                    image_embeddings = np.load(image_embedding_path)

                    object_1_embeddings = object_embeddings[object_id_to_embeddings_index[box_1.class_id]]
                    if len(object_1_embeddings) > 1:
                        object_1_embeddings = np.expand_dims(object_1_embeddings[0], axis=0)
                    object_2_embeddings = object_embeddings[object_id_to_embeddings_index[box_2.class_id]]
                    if len(object_2_embeddings) > 1:
                        object_2_embeddings = np.expand_dims(object_2_embeddings[0], axis=0)
                    objects_embeddings = np.stack((object_1_embeddings, object_2_embeddings), axis=1)

                    feed_dict = {
                        model.image_embeddings: np.array([image_embeddings]),
                        model.object_embeddings: objects_embeddings
                    }

                    results = model.sess.run(model.predictions, feed_dict=feed_dict)

                    print(results[0])

                f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_predictions', required=True)
    parser.add_argument('--class_ids_path', default=r'X:\OpenImages\rel\challenge-2018-classes-vrd.csv')
    parser.add_argument('--relationships_ids_path', default=r'X:\OpenImages\rel\labels.txt')

    parser.add_argument('--image_embeddings_dir', default=r'X:\OpenImages\embeddings\pnasnet_large\challenge2018')
    parser.add_argument('--image_embedding_size', type=int, default=4320)

    parser.add_argument('--objects_embeddings_path', default=r'X:\OpenImages\embeddings\words_lm_1b\embeddings_char_cnn.npy')
    parser.add_argument('--vocab_path', default=r'X:\OpenImages\rel\vocab_with_object_ids.txt')

    parser.add_argument('--checkpoints_dir', type=str, required=True)

    parser.add_argument('--images_dir', type=str)

    main(parser.parse_args())
