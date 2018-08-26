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

from triplets.submission.utils import parse_prediction_line


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
    return list(permutations(boxes))


def build_permutations(input_predictions, class_ids):
    all_permutations = []
    for l in input_predictions:
        boxes_data = l[1]
        boxes = parse_prediction_line(boxes_data)
        boxes = filter_relevant_class_ids(class_ids=class_ids, boxes=boxes)
        boxes = filter_out_duplicate_class_ids(boxes)
        perms = build_box_permutations(boxes)
        all_permutations += [(l[0], perms)]
    return all_permutations


def main(args):

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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_predictions', default=r'D:\Projects\OpenImagesChallenge\keras-yolo3\submission\merged\2018-08-24_20-10-00.csv')
    parser.add_argument('--class_ids_path', default=r'X:\OpenImages\rel\challenge-2018-classes-vrd.csv')
    parser.add_argument('--relationships_ids_path', default=r'X:\OpenImages\rel\labels.txt')

    parser.add_argument('--image_embeddings_dir', type=str, required=True)
    parser.add_argument('--image_embedding_size', type=int, default=4320)

    parser.add_argument('--objects_embeddings_path', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)

    parser.add_argument('--checkpoints_dir', type=str, required=True)

    parser.add_argument('--images_dir', type=str)

    main(parser.parse_args())
