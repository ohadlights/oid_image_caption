import argparse
from tqdm import tqdm


def process_file(val_ids, source_path):
    output_train = source_path.replace('.csv', '_train.csv')
    output_val = source_path.replace('.csv', '_val.csv')

    content = [l.strip().split(',') for l in open(source_path).readlines()]
    with open(output_train, 'w') as ft, open(output_val, 'w') as fv:
        for l in tqdm(content):
            image_id = l[0]
            f = fv if image_id in val_ids else ft
            f.write('{}\n'.format(','.join(l)))


def main(args):
    val_ids = set([l.strip() for l in open(args.val_ids_path).readlines()])
    process_file(val_ids, args.labels_csv_path)
    process_file(val_ids, args.bbox_csv_path)
    process_file(val_ids, args.relationships_csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_ids_path', required=True)
    parser.add_argument('--labels_csv_path', required=True)
    parser.add_argument('--bbox_csv_path', required=True)
    parser.add_argument('--relationships_csv_path', required=True)
    main(parser.parse_args())
