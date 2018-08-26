import argparse
import tensorflow as tf
from triplets.data_provider import DataProvider
from triplets.model_v1 import ModelV1
from triplets.evaluate_triplets_model import evaluate_model


def main(args):

    # data

    data_provider = DataProvider(args.annotations_csv_path,
                                 args.batch_size,
                                 args.image_embeddings_dir,
                                 args.objects_embeddings_path,
                                 args.vocab_path,
                                 embedding_extension='.jpg.npy')
    object_embedding_size = data_provider.get_object_embedding_size()
    num_classes = data_provider.get_num_classes()

    # Build model

    with ModelV1(image_embedding_size=args.image_embedding_size,
                 object_embedding_size=object_embedding_size,
                 num_classes=num_classes,
                 checkpoints_dir=args.checkpoints_dir) as model:

        # evaluate

        evaluate_model(model.predictions,
                       image_embeddings=model.image_embeddings,
                       object_embeddings=model.object_embeddings,
                       data_provider=data_provider,
                       sess=model.sess,
                       batch_size=args.batch_size,
                       max_samples=args.max_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_csv_path', type=str, required=True)
    parser.add_argument('--image_embeddings_dir', type=str, required=True)
    parser.add_argument('--images_dir', type=str)
    parser.add_argument('--objects_embeddings_path', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--checkpoints_dir', type=str, required=True)

    parser.add_argument('--image_embedding_size', type=int, default=4320)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_samples', type=int, default=100000)

    main(parser.parse_args())