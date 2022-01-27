import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
import subprocess


class DataPrep:
    """
    Prep data to create nodes, coordinates and correlations with indices
    """

    def __init__(self, img_data, path, edges_file=None, node_file=None, node_file_txt=None, coord_file=None):


        if isinstance(img_data, np.ndarray):
            vox, shape = img_data, img_data.shape
        elif isinstance(img_data, nib.nifti1.Nifti1Image):
            img = img_data
            vox, shape = self.voxels(img)
        else:
            raise

        if os.path.isdir(os.path.join(path)):
            self.path = os.path.join(path)
        else:
            raise EnvironmentError('no data subdirectory found')

        if edges_file:
            if isinstance(edges_file, str):
                if '.npy' in edges_file:
                    self.edges_file = edges_file
                else:
                    raise NameError('Edges file must be .npy file (.npy)')
            else:
                raise TypeError('Edges filename must be a string')
        else:
            self.edges_file = None

        if node_file:
            if isinstance(node_file, str):
                if '.npy' in node_file:
                    self.node_file = node_file
                else:
                    raise NameError('Node_list file must be Numpy Array (.npy)')
            else:
                raise TypeError('Node_list filename must be a string')
        else:
            self.node_file = None

        if node_file_txt:
            if isinstance(node_file_txt, str):
                if '.txt' in node_file_txt:
                    self.node_file_txt = node_file_txt
                else:
                    raise NameError('node_list file must be Text File (.txt)')
            else:
                raise TypeError('node_list filename must be a string')
        else:
            self.node_file_txt = None

        if coord_file:
            if isinstance(coord_file, str):
                if '.npy' in coord_file:
                    self.coord_file = coord_file
                else:
                    raise NameError('coordinate dictionary (coords) file must be Numpy Array (.npy)')
            else:
                raise TypeError('coordinate dictionary (coords) filename must be a string')
        else:
            self.coord_file = None

        nodes = self.time_series(vox)
        n_nodes = self.zfilter(nodes)

        self.coords = n_nodes[:,:3]
        self.n_nodes = n_nodes[:,3:103]

        self.num_vox = int(self.n_nodes.shape[0])
        self.num_com = int(((self.num_vox ** 2) - self.num_vox) // 2)

    def nodes(self):
        """
        return nodes and save if desired
        :return: node array of shape [num_nodes, node feature vector length]
        """

        if self.node_file:
            np.save(os.path.join(self.path, self.node_file), arr=np.asarray(self.n_nodes), allow_pickle=True)
            np.savetxt(os.path.join('/home/ubuntu', self.node_file_txt), self.n_nodes)
        return self.n_nodes.shape

    def coordinates(self):
        """
        return coords and save if desired
        :return: list array of shape {Node #: [coordinates]}
        """
        if self.coord_file:
            np.save(os.path.join(self.path, self.coord_file), arr=np.asarray(self.coords), allow_pickle=True)
        return self.coords


    def generate_corrs(self):
        os.system('nvcc -lcublas -O2 -arch=compute_50 -code=sm_50 -std=c++11 ./Fast-GPU-PCC/CPU_side.cpp ./Fast-GPU-PCC/GPU_side.cu -o exec')
        sh1 = self.n_nodes.shape[0]
        sh2 = self.n_nodes.shape[1]
        os.system('./exec sh1 sh2 b')

        cmd = ['./exec', str(sh1), str(sh2), 'b']
        subprocess.call(cmd)

        # cmd2 = ['mv', 'corrs.bin', self.path]
        # subprocess.call(cmd2)

    def edge_indices(self):
        """
        return edges and save if desired
        :return: edgelist array of shape [num_edges, 2] where axis 1 is [tuple(node_x, node_y), correlation_coefficient]
        """

        combs = self.np_combinations(self.num_vox,2)
        indices = combs.T

        corrs = np.fromfile('/home/ubuntu/corrs.bin', dtype=np.float32)
        corrs = np.reshape(corrs, (corrs.shape[0], 1))
        print(corrs.shape)
        corrs1 = corrs[abs(corrs) > 0.9]
        corrs1 = np.reshape(corrs1, (corrs1.shape[0], 1))
        print(corrs1.shape)

        indices1 = indices[:,0]
        indices1 = np.reshape(indices1, (indices1.shape[0], 1))
        print(indices1.shape)
        indices1 = indices1[abs(corrs) > 0.9]
        indices1 = np.reshape(indices1, (indices1.shape[0], 1))
        print(indices1, indices1.shape)

        indices2 = indices[:,1]
        indices2 = np.reshape(indices2, (indices2.shape[0], 1))
        print(indices2.shape)
        indices2 = indices2[abs(corrs) > 0.9]
        indices2 = np.reshape(indices2, (indices2.shape[0], 1))
        print(indices2, indices2.shape)

        edges = np.concatenate((indices1, indices2, corrs1), axis = 1)

        if self.edges_file:

            np.save(os.path.join(self.path, self.edges_file), edges, allow_pickle=True)

        # path3 = os.path.join(self.path, 'corrs.bin')
        # cmd3 = ['rm', path3]
        # subprocess.call(cmd3)
        os.system('rm ./corrs.bin')
        os.system('rm ./data.txt')

        return edges

    @staticmethod
    def voxels(image):
        """
        Convert nii image to numpy array of voxels
        :param image: dataobj of nii file
        :return: numpy array, shape of numpy array
        """
        voxels = np.array(image.dataobj)
        shape = voxels.shape
        return voxels, shape

    @staticmethod
    def time_series(voxels):
        """
        Convert 3d numpy array to flattened array
        :param voxels: numpy array
        :return: flattened array
        """
        t_axis = []
        coords = {}
        for o, a in enumerate(voxels):
            for m, b in enumerate(a):
                for i, c in enumerate(b):
                    t_axis.append(c)
                    coords[(int(len(t_axis) - 1))] = [o, m, i]
        coords  = np.array(coords)
        nodes = [np.append(np.asarray(coords[()][i]), x, axis=None) for i, x in enumerate(t_axis)]
        return np.array(nodes)

    @staticmethod
    def zfilter(nodes: np.array):
        """
        Filter voxels that have no signal
        :param nodes: Numpy array of shape (number nodes, node time length)
        :return: Numpy array with voxels that have a signal
        """
        return nodes[~np.all(nodes[:,3:] == 0, axis=1)]

    @staticmethod
    def np_combinations(n, k): #n is the number of nodes, k is the length of a combination (2)
        a = np.ones((k, n - k + 1), dtype=int)
        a[0] = np.arange(n - k + 1)
        for j in range(1, k):
            reps = (n - k + j) - a[j - 1]
            a = np.repeat(a, reps, axis=1)
            ind = np.add.accumulate(reps)
            a[j, ind[:-1]] = 1 - reps[1:]
            a[j, 0] = j
            a[j] = np.add.accumulate(a[j])
        return a



# if __name__ == '__main__':
#     data_actual = nib.load(os.path.join('/home/ubuntu/ABIDE/Pitt_0050033_func_preproc.nii.gz'))
#     # data_fake = np.random.randn(5, 5, 5, 100)
#
#     V = DataPrep(data_actual,
#                  edges_file='edges.npy',
#                  node_file='nodes.npy',
#                  node_file_txt = 'data.txt',
#                  coord_file='coordinates.npy')
#
#
#     V.nodes()
#
#     V.coordinates()
#
#     V.generate_corrs()
#
#     V.edge_indices()


root = '/home/ubuntu/ADHD200/Brown'
for dir in os.listdir(root):
    try:
        if os.path.exists(os.path.join(root, dir, str(dir) +'_edges09.npy')):
            pass
        else:
            path = os.path.join(root, dir)
            fmri = 'fmri_' + str(dir) + '.nii.gz'
            data_actual = nib.load(os.path.join(path, fmri))
            print(os.path.join(path, fmri))

            V = DataPrep(data_actual,
                         path,
                         edges_file= str(dir) +'_edges09.npy',
                         node_file= str(dir) + '_nodes.npy',
                         node_file_txt='data.txt',
                         coord_file= str(dir)+'_coordinates.npy')
            V.nodes()
            V.coordinates()
            V.generate_corrs()
            V.edge_indices()
    except Exception:
        pass




