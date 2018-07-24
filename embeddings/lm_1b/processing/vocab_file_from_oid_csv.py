import argparse


def process_vocab(words_set, path):
    content = open(path).readlines()
    content = [l.strip().split(',') for l in content]
    for entry in content:
        id = entry[0]
        word = entry[1]
        words = word.replace('(', '').replace(')', '').split()
        for w in words:
            if w not in words_set:
                words_set[w] = id


def main(args):
    classes_vocab = args.words_vocab
    attributes_vocab = args.attributes_vocab
    relationships_vocab = args.relationships_vocab
    output_path = args.output_path

    vocabs = [classes_vocab, attributes_vocab, relationships_vocab]
    words_set = {'<S>': '<S>', '</S>': '</S>'}

    for v in vocabs:
        process_vocab(words_set, v)

    with open(output_path, 'w') as f:
        content = open(r'D:\Projects\OpenImagesChallenge\image-caption\embeddings\lm_1b\files\original_words_list_for_embeddings_order.txt').readlines()
        for l in content:
            w = l.strip()
            id = words_set[w]
            f.write('{},{}\n'.format(w, id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('words_vocab')
    parser.add_argument('attributes_vocab')
    parser.add_argument('relationships_vocab')
    parser.add_argument('--output_path', default=r'..\files\oid_vocab.txt')
    main(parser.parse_args())