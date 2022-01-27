import torch_geometric as tg
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.data import Data
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sklearn

#
# class BrainDataset(InMemoryDataset):
#     def __init__(self, root):
#         super(BrainDataset, self).__init__(root=root)
#
#         self.data, self.slices = torch.load(self.processed_paths[0])
#         self.root = root
#
#
#     @property
#     def raw_file_names(self):
#         return ['_edges.npy', '_nodes.npy', '_coordinates.npy']
#
#     @property
#     def processed_file_names(self):
#         return ['processed.pt']
#
#     def _download(self):
#         pass
#
#     def process(self):
#         """
#         :return:
#         """
#
#         data_list = []
#
#         for dir in tqdm(os.listdir(self.root)):
#             path = os.path.join(self.root, dir)
#
#             subject_ID = dir
#             df = pd.read_csv('/home/ubuntu/abidegraphs/Phenotypic_V1_0b_preprocessed1.csv')
#             d = df.loc[df['FILE_ID'] == subject_ID, 'DX_GROUP']
#             d = d.to_numpy()
#             print(dir)
#             print(d)
#
#             edges = np.load(os.path.join(path, dir + self.raw_file_names[0]), allow_pickle=True)
#             nodes = np.load(os.path.join(path, dir + self.raw_file_names[1]), allow_pickle=True)
#             coords = np.load(os.path.join(path, dir + self.raw_file_names[2]), allow_pickle=True)
#
#             if len(nodes[0]) < 100:
#                 nodes = np.pad(nodes, pad_width=((0,0),(0,100-(len(nodes[0])))))
#
#             # edges, nodes, coords = [np.load(x, allow_pickle=True) for x in self.raw_paths]
#
#
#             # add the coordinates to the node feature vector
#             nodes = [np.append(np.asarray(coords[()][i]), x, axis=None) for i, x in enumerate(nodes)]
#
#             # PyG Requires non-object numpy arrays
#             nodes = np.asarray(nodes, dtype=np.float32)
#             nodes = torch.from_numpy(nodes)
#             print(nodes.shape)
#
#             edges = np.asarray(edges, dtype=np.float32)
#             # edges = [x for x in tqdm(edges) if x[2] > 0.9]
#             edges = np.asarray(edges)
#
#             edge_indices = edges[:,:2]
#             edge_indices = edge_indices.astype(int)
#             edge_indices = torch.from_numpy(edge_indices)
#             edge_indices = np.transpose(edge_indices)
#             print(edge_indices.shape)
#             print(edge_indices[:,10])
#
#
#             edge_attr=edges[:,2]
#             edge_attr = torch.from_numpy(edge_attr)
#             print(edge_attr.shape)
#             print(edge_attr[10])
#
#
#             # y = torch.tensor([1], dtype=int)
#             y =torch.from_numpy(d)
#
#             # Convert to pytorch geometric object
#             data = Data(x = nodes , edge_index = edge_indices , edge_attr = edge_attr, y = y)
#
#             data_list.append(data)
#
#             data, slices = self.collate(data_list = data_list)
#             torch.save((data, slices), self.processed_paths[0])
#
#         return data_list


class BrainDataset2(Dataset):
    def __init__(self, root):
        super(BrainDataset2, self).__init__(root=root)
        self.root = root

    @property
    def raw_file_names(self):
        return ['_edges.npy', '_nodes.npy', '_coordinates.npy']

    @property
    def processed_file_names(self):
        i = 0
        file_names = []
        for dir in os.listdir(self.root):
            if i < 969:
                file_names.append('data_{}.pt'.format(i))
                i += 1
        return file_names

    @property
    def processed_dir(self):
        processed_dir = '/home/ubuntu/abidegraphs/processed/'
        return processed_dir

    @property
    def num_classes(self):
        """The number of classes in the dataset."""
        return len(np.unique(self.y))


    def _download(self):
        pass

    def process(self):
        """
        :return:
        """
        y_vals = []
        i = 0
        for dir in tqdm(os.listdir(self.root)):
            path = os.path.join(self.root, dir)

            if dir.startswith('00'):
                subject_ID = dir[2:]
            else:
                subject_ID = dir


            print(dir)
            print(subject_ID)

            if dir == 'Phenotypic_V1_0b_preprocessed1.csv' or dir == 'processed':
                pass
            else:
                df = pd.read_csv('/home/ubuntu/abidegraphs/Phenotypic_V1_0b_preprocessed1.csv')
                d = df.loc[df['FILE_ID'] == subject_ID, 'DX_GROUP']
                d = (d.to_numpy())-1

                print(dir)
                print(d)


                edges = np.load(os.path.join(path, dir + self.raw_file_names[0]), allow_pickle=True)
                nodes = np.load(os.path.join(path, dir + self.raw_file_names[1]), allow_pickle=True)
                coords = np.load(os.path.join(path, dir + self.raw_file_names[2]), allow_pickle=True)

                # nodes = np.transpose(nodes)
                # scaler = sklearn.preprocessing.StandardScaler()
                # nodes = scaler.fit_transform(nodes)
                # nodes = np.transpose(nodes)

                nodes = np.round_(nodes, 2)
                print(nodes)

                if len(nodes[0]) < 100:
                    nodes = np.pad(nodes, pad_width=((0,0),(0,100-(len(nodes[0])))))

                # edges, nodes, coords = [np.load(x, allow_pickle=True) for x in self.raw_paths]


                # add the coordinates to the node feature vector
                nodes = [np.append(np.asarray(coords[()][i]), x, axis=None) for i, x in enumerate(nodes)]

                # PyG Requires non-object numpy arrays
                nodes = np.asarray(nodes, dtype=np.float32)
                nodes = torch.from_numpy(nodes)
                print(nodes.shape)

                edges = np.asarray(edges, dtype=np.float32)
                edges = [x for x in tqdm(edges) if x[2] > 0.9]
                edges = np.asarray(edges)

                edge_indices = edges[:,:2]
                edge_indices = edge_indices.astype(int)
                edge_indices = torch.from_numpy(edge_indices)
                edge_indices = np.transpose(edge_indices)
                print(edge_indices.shape)
                print(edge_indices[:,10])


                edge_attr = np.round_(edges[:,2],2).abs()
                # edge_attr = np.reshape(edge_attr, len(edge_attr))
                edge_attr = torch.from_numpy(edge_attr)
                print(edge_attr.shape)
                print(edge_attr[10])


                y = torch.tensor(d)
                print(y)
                # y = torch.from_numpy(d)
                y_vals.append(d)

                # Convert to pytorch geometric object
                data = Data(x = nodes , edge_index = edge_indices , edge_attr = edge_attr, y = y)

                torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1
        self.i = i
        self.y = y_vals

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data




