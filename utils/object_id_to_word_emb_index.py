import argparse


def main(args):
    content = [l.strip() for l in open(args.words_vocab_path).readlines()]
    word_to_index = {content[i]: str(i) for i in range(len(content))}

    object_descs = [l.strip().split(',') for l in open(args.objects_desc_path).readlines()]
    object_descs += [l.strip().split(',') for l in open(args.attributes_desc_path).readlines()]

    with open(args.output_path, 'w') as f:

        for l in object_descs:

            id = l[0]
            words = l[1].replace('(', '').replace(')', '').split()

            indexes = []
            for w in words:
                if w in word_to_index:
                    indexes += [word_to_index[w]]
                else:
                    print('N/A {}'.format(w))

            f.write('{},{}\n'.format(id, ' '.join(indexes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--words_vocab_path', type=str, required=True)
    parser.add_argument('--objects_desc_path', type=str, required=True)
    parser.add_argument('--attributes_desc_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    main(parser.parse_args())
