import sys
import cv2
import torch
import sparse
import pickle
import random
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image, ImageFile
from torch_utils import persistence
from torchvision import transforms
from wilds.datasets.wilds_dataset import WILDSDataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append('.')


@persistence.persistent_class
class STAugmentPipe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Init unused p for compatibility
        self.register_buffer('p', torch.ones([]))
        self.flip = transforms.RandomHorizontalFlip()

    def forward(self, img):
        assert isinstance(img, torch.Tensor) and img.ndim == 4
        img = torch.rot90(img, random.randint(0, 3), [2, 3])
        img = self.flip(img)
        return img


class STDataset(WILDSDataset):
    def __init__(self,
                 data,
                 gene_num,
                 gene_spa=False,
                 root_dir=Path('Data'),
                 transform=None,
                 split_scheme=None,
                 debug=False,
                 seed=None,
                 # compatible to stylegan3
                 resolution=128,
                 num_channels=3,
                 max_size=None):

        self._dataset_name = data
        self.gene_num = gene_num
        self.gene_spa = gene_spa
        self.img_dir = root_dir / f'{data}/GAN/crop'
        self.trans = transform
        # This is for compatible to stylegan3 training
        self.debug = debug
        if seed:
            random.seed(seed)
        self.resolution = resolution
        self.num_channels = num_channels

        # Prep metadata including cropped image paths
        df = pd.read_csv(str(self.img_dir / 'metadata.csv'))
        self._input_array = df.path.values
        self.ext = self._input_array[0].split('_')[-1]
        print(f'Path extension of {data}: {self.ext}')

        # Prep genedata img if exists
        self.expr_img = self.prep_gene(self.img_dir / 'metadata_img.csv',
                                       True)

        # Prep genedata cell if exists
        self.expr_cell = self.prep_gene(self.img_dir / 'metadata_cell.csv',
                                        False)

        # Prep subsetdata for downstream analysis
        self.prep_split(df, split_scheme)

    def prep_gene(self, gene_pth, load_name=True):
        if not self.debug:
            assert gene_pth.is_file()

        if gene_pth.is_file():
            _expr = pd.read_csv(str(gene_pth),
                                index_col=0)
            if self._dataset_name in ('CosMx', 'Xenium'):
                _expr = _expr.astype(np.int16)
            elif self._dataset_name == 'Visium':
                _expr = _expr.astype(np.float32)
            _expr = _expr.to_numpy()
        else:
            _expr = None
            print(f'{str(gene_pth)} does not exist')

        # list of gene names
        if load_name:
            with open(str(self.img_dir / 'transcripts.pickle'), 'rb') as fp:
                self.gene_name = pickle.load(fp)
            assert self.gene_num == len(self.gene_name)

        return _expr

    def prep_split(self, df, split_scheme):
        if split_scheme is not None:
            t_nam, t_cnt = np.unique(df[split_scheme].values,
                                     return_counts=True)
            # if the counts have repetitive values, then it cannot be used
            # for stratify subset, thus add another _cond_dct
            self._cond_dct = dict(zip(t_nam, t_cnt))
            if len(t_nam) == 2:
                # this is for GAN training when including dichotomy domain label
                self._split_dict = dict(zip(t_nam, [-1, 1]))
            else:
                self._split_dict = dict(zip(t_nam, list(range(len(t_nam)))))
            _y = df[split_scheme].map(self._split_dict).values
            self._split_array = _y

            self._metadata_fields = [split_scheme, ]
            self._metadata_array = list(zip(*[self._split_array.tolist()]))
            self._y_array = torch.LongTensor(
                (_y + 1) // 2 if len(t_nam) == 2 else _y)
            self._y_size, self._n_classes = 1, len(t_nam)
        else:
            # add dummy metadata amd labels
            self._metadata_array = [[0] for _ in range(len(self._input_array))]
            self._y_array = torch.ones([len(self._input_array)]).long()
            self._y_size, self._n_classes = 1, 2

    def get_input(self, idx):
        img_pth = self.img_dir / self._input_array[idx]
        gene_pth = str(img_pth).replace(self.ext, 'rna.npz')

        if not self.debug:
            img, gene_expr = self.run_input(img_pth, gene_pth, idx)
        else:
            img = torch.empty([0, 128, 128])
            if self.expr_img is not None:
                # Here img is the processed gene img and gene cell cmp
                img, gene_expr = self.run_debug(
                    self.expr_img[idx], gene_pth, idx)
            else:
                gene_expr = sparse.load_npz(gene_pth)
                # This part is for creating metadata_img.csv
                if self._dataset_name in ('CosMx', 'Xenium'):
                    gene_expr = gene_expr.sum((0, 1)).todense()
                    gene_expr = gene_expr.astype(np.int16)

        return img, gene_expr

    def run_input(self, img_pth, gene_pth, idx):
        img = np.array(Image.open(img_pth))
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).contiguous().float()

        if self.gene_spa:
            assert self._dataset_name in ('CosMx', 'Xenium')
            gene_expr = sparse.load_npz(gene_pth)
            # img = torch.from_numpy(gene_expr.todense().transpose((2, 0, 1)))
            if self.trans is not None:
                img, gene_expr = self.trans([img, gene_expr])

            # # naive init sparse tensor, incompatible to spconv 
            # i = torch.LongTensor(np.array(gene_expr.coords))
            # v = torch.FloatTensor(np.array(gene_expr.data))
            # s = torch.Size(gene_expr.shape)
            # gene_expr = torch.sparse.FloatTensor(i, v, s)

            # init sparse tensor compatible to spconv
            i = torch.LongTensor(np.array(gene_expr.coords[:-1]))
            # create channel matrix indices
            c = gene_expr.coords[-1]
            r = list(range(len(c)))
            v = torch.zeros([len(c), self.gene_num]).float()
            # the data v is nothing but the matrix that 
            # each row correspondes to the gene_num of readouts at certain spatial loc
            v[r, c] = torch.from_numpy(gene_expr.data.astype(np.float32))
            s = torch.Size(gene_expr.shape)
            gene_expr = torch.sparse_coo_tensor(i, v, s)
            

            # gcmp = torch.from_numpy(self.expr_img[idx]).contiguous().float()
            # gcmp1 = gene_expr.to_dense().sum((0, 1))
            # gcmp2 = gene_expr.coalesce().values().sum((0))
            # assert (gcmp == gcmp1).all() and (gcmp == gcmp2).all() 
            # assert((img == gene_expr.to_dense().permute((2, 0, 1))).all())

            return img, gene_expr
        else:
            gene_expr = self.expr_img[idx]
            gene_expr = torch.from_numpy(gene_expr).contiguous().float()

            # append label for stylegan3 conditional training
            label = torch.nn.functional.one_hot(self._y_array[idx],
                                                num_classes=self._n_classes)
            gene_expr = torch.cat([gene_expr, label])

            if self.trans is not None:
                img = self.trans(img)
            return img, gene_expr

    def run_debug(self, gene_expr, gene_pth, idx):
        out = sparse.load_npz(gene_pth).todense()
        if self._dataset_name in ('CosMx', 'Xenium'):
            assert (out.sum((0, 1)) == gene_expr).all()
            if self._dataset_name == 'CosMx':
                cell_pth = gene_pth.replace('rna.npz', 'cell.png')
                cell_np = cv2.imread(cell_pth, flags=cv2.IMREAD_UNCHANGED)
                # dir/Liver1/c_1_10_100_rna.npz, cid = 100
                cid = int(Path(gene_pth).name.split('_')[-2])
                out[cell_np != cid] = 0
            else:
                nucl_pth = gene_pth.replace('rna.npz', 'nucl.npz')
                nucl_coo = sparse.load_npz(nucl_pth).todense()
                cell_pth = gene_pth.replace('rna.npz', 'cell.npz')
                cell_coo = sparse.load_npz(cell_pth).todense()
                # dir/Rep1/*_*_1234_*_*_*_*_*_*_*_*_rna.npz, cid = 1234
                cid = int(Path(gene_pth).name.split('_')[2])
                out[(nucl_coo != cid) & (cell_coo != cid)] = 0
            out = np.abs(out.sum((0, 1)) -
                         self.expr_cell[idx])
        else:
            assert (out == gene_expr).all()
        return out, gene_expr