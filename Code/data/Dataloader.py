import torch_geometric as tg
import torch

class BrainDataset(tg.data.InMemoryDataset):
    def __init__(self, root):
        super(BrainDataset, self).__init__(root=root)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['actual_correlations.npy']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _download(self):
        pass

    def process(self):
        """
        task options are quaternary, binary, teleological
        :return:
        """

        data_fake = np.random.randn(5, 5, 5, 100)
        V = VoxelCorrelations(data_fake, cutoff=0.09)
        G = BrainGraph(nodes=V.nodes(), edges=V.edges())
        T = G.pyg()
        data = T

        data_list = [T]
        data, slices = self.collate(data_list=data_list)
        torch.save((data, slices), self.processed_paths[0])