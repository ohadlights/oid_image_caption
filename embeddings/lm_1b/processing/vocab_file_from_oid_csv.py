import argparse


def process_vocab(words_set, path):
    content = open(path).readlines()
    content = [l.strip().split(',')[1] for l in content]
    for entry in content:
        words = entry.replace('(', '').replace(')', '').split()
        for w in words:
            words_set.add(w)


def main(args):
    classes_vocab = args.words_vocab
    attributes_vocab = args.attributes_vocab
    relationships_vocab = args.relationships_vocab
    output_path = args.output_path

    vocabs = [classes_vocab, attributes_vocab, relationships_vocab]
    words_set = {'<S>', '</S>'}

    for v in vocabs:
        process_vocab(words_set, v)

    with open(output_path, 'w') as f:
        for w in words_set:
            f.write('{}\n'.format(w))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('words_vocab')
    parser.add_argument('attributes_vocab')
    parser.add_argument('relationships_vocab')
    parser.add_argument('--output_path', default=r'..\files\oid_vocab.txt')
    main(parser.parse_args())