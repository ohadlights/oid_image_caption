import os
import random
import numpy as np


class DataProvider:
    def __init__(self, annotations_csv_path, batch_size, image_embeddings_dir, objects_embeddings_path, vocab_path):
        self.batch_size = batch_size
        self.image_embeddings_dir = image_embeddings_dir
        self.current_index = 0

        content = [l.strip().split(',') for l in open(annotations_csv_path).readlines()]
        data = []
        for a in content:
            image_id = a[0]
            object_1 = a[1]
            object_2 = a[2]
            label = int(a[-1])
            data += [(image_id, object_1, object_2, label)]
        self.data = data

        self.object_embeddings = np.load(objects_embeddings_path)

        content = [l.strip().split(',') for l in open(vocab_path).readlines()]
        self.object_id_to_embeddings_index = {content[i][1]: i for i in range(len(content))}

    def next_epoch(self):
        random.shuffle(self.data)
        self.current_index = 0

    def next_batch(self):
        image_embeddings = []
        objects_embeddings = []
        labels = []
        for i in range(self.current_index, self.current_index+self.batch_size):
            data = self.data[i]

            image_id = data[0]
            image_embedding_path = os.path.join(self.image_embeddings_dir, image_id + '.jpg.npy')
            image_embeddings += np.load(image_embedding_path)

            object_1_emb = self.object_embeddings[self.object_id_to_embeddings_index[data[1]]]
            object_2_emb = self.object_embeddings[self.object_id_to_embeddings_index[data[2]]]
            objects_embeddings += [np.concatenate((object_1_emb, object_2_emb))]

            labels += data[3]

        return np.array(image_embeddings), np.array(objects_embeddings), np.array(labels)
