import os
import random
import numpy as np


class DataProvider:
    def __init__(self, annotations_csv_path, batch_size, image_embeddings_dir, objects_embeddings_path, vocab_path):
        self.batch_size = batch_size
        self.image_embeddings_dir = image_embeddings_dir
        self.current_index = 0

        content = [l.strip().split(',') for l in open(annotations_csv_path).readlines()[1:]]
        data = []
        labels = set()
        for a in content:
            image_id = a[0]
            object_1 = a[1]
            object_2 = a[2]
            label = int(a[-1])
            data += [(image_id, object_1, object_2, label)]
            labels.add(label)
        self.data = data
        self.labels = labels

        self.object_embeddings = np.load(objects_embeddings_path).astype(np.float32)

        content = [l.strip().split(',') for l in open(vocab_path).readlines()]
        self.object_id_to_embeddings_index = {content[i][0]: [int(a) for a in content[i][1].split()] for i in range(len(content))}

        random.seed(42)

    def next_epoch(self):
        random.shuffle(self.data)
        self.current_index = 0

    def next_batch(self):
        image_embeddings = []
        objects_embeddings = []
        labels = []
        for i in range(self.current_index, min(self.current_index+self.batch_size, len(self.data))):
            data = self.data[i]

            # image embeddings

            image_id = data[0]
            image_embedding_path = os.path.join(self.image_embeddings_dir, image_id + '.jpg.npy')
            image_embeddings += [np.load(image_embedding_path)]

            # object embeddings

            object_1_emb = None
            object_2_emb = None

            emb_indexes_1 = self.object_id_to_embeddings_index[data[1]]
            for index in emb_indexes_1:
                if object_1_emb is None:
                    object_1_emb = self.object_embeddings[index]
                else:
                    object_1_emb += self.object_embeddings[index]

            emb_indexes_2 = self.object_id_to_embeddings_index[data[2]]
            for index in emb_indexes_2:
                if object_2_emb is None:
                    object_2_emb = self.object_embeddings[index]
                else:
                    object_2_emb += self.object_embeddings[index]

            objects_embeddings += [np.stack((object_1_emb, object_2_emb))]

            # labels

            labels += [data[3]]

            # increment index

            self.current_index += 1

        return np.array(image_embeddings), np.array(objects_embeddings), np.array(labels)

    def get_object_embedding_size(self):
        return self.object_embeddings.shape[1]

    def get_num_classes(self):
        return max(self.labels) + 1

    def get_num_examples(self):
        return len(self.data)
