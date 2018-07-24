import os
import argparse
from tqdm import tqdm


def create_labels_list(source_path):
    output_path = os.path.join(os.path.dirname(source_path), 'labels.txt')
    labels = list(set([l.strip().split(',')[-1] for l in open(source_path).readlines()]))
    label_str_to_index = {labels[i]: i for i in range(len(labels))}
    with open(output_path, 'w') as f:
        for k, v in label_str_to_index.items():
            f.write('{},{}\n'.format(k, v))
    return label_str_to_index


def process_file(val_ids, source_path, label_str_to_index=None):
    output_train = source_path.replace('.csv', '_train.csv')
    output_val = source_path.replace('.csv', '_val.csv')

    content = [l.strip().split(',') for l in open(source_path).readlines()]
    with open(output_train, 'w') as ft, open(output_val, 'w') as fv:
        for l in tqdm(content):
            if label_str_to_index:
                l += [str(label_str_to_index[l[-1]])]
            image_id = l[0]
            f = fv if image_id in val_ids else ft
            f.write('{}\n'.format(','.join(l)))


def main(args):
    label_str_to_index = create_labels_list(args.relationships_csv_path)

    val_ids = set([l.strip() for l in open(args.val_ids_path).readlines()])

    process_file(val_ids, args.labels_csv_path)
    process_file(val_ids, args.bbox_csv_path)
    process_file(val_ids, args.relationships_csv_path, label_str_to_index=label_str_to_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_ids_path', required=True)
    parser.add_argument('--labels_csv_path', required=True)
    parser.add_argument('--bbox_csv_path', required=True)
    parser.add_argument('--relationships_csv_path', required=True)
    parser.add_argument('--labels_list_path', required=True)
    main(parser.parse_args())
