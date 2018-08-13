import os
import argparse
from tqdm import tqdm


'''
000002b66c9c498e.jpg.npy
000002b66c9c498e.jpg.0798.npy
'''

def main(args):
    per_image = dict()
    annotations = open(args.annotations_path).readlines()
    header = annotations[0].strip()
    annotations = [l.strip().split(',') for l in annotations[1:]]
    for l in tqdm(annotations):
        id = l[0]
        per_image[id] = l[1:]

    files = os.listdir(args.embeddings_dir)
    with open(args.output_path, 'w') as f:
        f.write('{}\n'.format(header))
        for file in tqdm(files):
            id = file.strip().split('.')[0]
            if id in per_image:
                a = per_image[id]
                f.write('{},{}\n'.format(file, ','.join(a)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_path', default=r'X:\OpenImages\rel\challenge-2018-train-vrd_train.csv')
    parser.add_argument('--output_path', default=r'X:\OpenImages\rel\challenge-2018-train-vrd_train_extended.csv')
    parser.add_argument('--embeddings_dir', default=r'X:\OpenImages\embeddings\pnasnet_large\train')
    main(parser.parse_args())