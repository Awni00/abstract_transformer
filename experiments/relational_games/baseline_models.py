import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys; sys.path.append('../..')
from original_abstractor_module import AbstractorModule


def create_patch_embedder(patch_height, patch_width, patch_dim, d_model):
    return nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, d_model),
                nn.LayerNorm(d_model),
        )

class CoRelNet(nn.Module):
    def __init__(self, image_shape, patch_size, d_model, num_classes):
        super(CoRelNet, self).__init__()

        self.img_channels, self.img_width, self.img_height = image_shape
        self.patch_width, self.patch_height = patch_size
        self.num_classes = num_classes
        self.d_model = d_model

        self.num_patches = (self.img_width // self.patch_width) * (self.img_height // self.patch_height)

        self.patch_dim = self.patch_width * self.patch_height * self.img_channels

        self.patch_embedder = create_patch_embedder(self.patch_height, self.patch_width, self.patch_dim, d_model)

        self.mlp_embedder = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))

        self.mlp_predictor = nn.Sequential(nn.Linear(self.num_patches**2, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.Linear(d_model, num_classes))

    def forward(self, x):
        x = self.patch_embedder(x)
        x = self.mlp_embedder(x)
        similarity_scores = torch.matmul(x, x.transpose(1, 2))
        x = torch.flatten(similarity_scores, start_dim=1)
        x = self.mlp_predictor(x)

        return x

# NOTE: temporary; forgot to include softmax in CoRelNet, CoRelNetSoftmax is "true" CoRelNet
class CoRelNetSoftmax(nn.Module):
    def __init__(self, image_shape, patch_size, d_model, num_classes):
        super(CoRelNetSoftmax, self).__init__()

        self.img_channels, self.img_width, self.img_height = image_shape
        self.patch_width, self.patch_height = patch_size
        self.num_classes = num_classes
        self.d_model = d_model

        self.num_patches = (self.img_width // self.patch_width) * (self.img_height // self.patch_height)

        self.patch_dim = self.patch_width * self.patch_height * self.img_channels

        self.patch_embedder = create_patch_embedder(self.patch_height, self.patch_width, self.patch_dim, d_model)

        self.mlp_embedder = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))

        self.mlp_predictor = nn.Sequential(nn.Linear(self.num_patches**2, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.Linear(d_model, num_classes))

    def forward(self, x):
        x = self.patch_embedder(x)
        x = self.mlp_embedder(x)
        similarity_scores = torch.matmul(x, x.transpose(1, 2))
        similarity_scores = torch.nn.functional.softmax(similarity_scores, dim=-1)
        x = torch.flatten(similarity_scores, start_dim=1)
        x = self.mlp_predictor(x)

        return x

class PrediNetModule(nn.Module):
    """PrediNet layer (Shanahan et al. 2020)"""

    def __init__(self, n_objects, d_model, n_relations, n_heads, key_dim=None, add_temp_tag=True):
        """create PrediNet layer.

        Parameters
        ----------
        key_dim : int
            key dimension
        n_heads : int
            number of heads (in PrediNet, this means the number of pairs of objects to compare)
        n_relations : int
            number of relations (in PrediNet, this means the number dimension of the difference relation)
        add_temp_tag : bool, optional
            whether to add temporal tag to object representations, by default False
        """
        super(PrediNetModule, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_relations = n_relations
        self.key_dim = key_dim if key_dim is not None else d_model // n_heads
        self.add_temp_tag = add_temp_tag

        if self.add_temp_tag:
            self.d_model += 1

        self.W_k = nn.Linear(self.d_model, self.key_dim, bias=False)
        self.W_q1 = nn.Linear(self.d_model * n_objects, self.n_heads * self.key_dim, bias=False)
        self.W_q2 = nn.Linear(self.d_model * n_objects, self.n_heads * self.key_dim, bias=False)
        self.W_s = nn.Linear(self.d_model, self.n_relations, bias=False)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()

    def forward(self, obj_seq):
        batch_size, n_objs, obj_dim = obj_seq.size()

        if self.add_temp_tag:
            temp_tag = torch.arange(n_objs, dtype=torch.float32, device=obj_seq.device).unsqueeze(0).unsqueeze(2)
            temp_tag = temp_tag.expand(batch_size, -1, -1)
            obj_seq = torch.cat([obj_seq, temp_tag], dim=2)


        obj_seq_flat = self.flatten(obj_seq)
        Q1 = self.W_q1(obj_seq_flat)
        Q2 = self.W_q2(obj_seq_flat)

        K = self.W_k(obj_seq)

        Q1_reshaped = Q1.view(batch_size, self.n_heads, self.key_dim)
        Q2_reshaped = Q2.view(batch_size, self.n_heads, self.key_dim)

        E1 = (self.softmax(torch.sum(Q1_reshaped.unsqueeze(1) * K.unsqueeze(2), dim=3))
              .unsqueeze(3) * obj_seq.unsqueeze(2))
        E1 = torch.sum(E1, dim=1)
        E2 = (self.softmax(torch.sum(Q2_reshaped.unsqueeze(1) * K.unsqueeze(2), dim=3))
              .unsqueeze(3) * obj_seq.unsqueeze(2))
        E2 = torch.sum(E2, dim=1)

        D = self.W_s(E1) - self.W_s(E2)

        if self.add_temp_tag:
            D = torch.cat([D, E1[:, :, -1].unsqueeze(2), E2[:, :, -1].unsqueeze(2)], dim=2)

        R = self.flatten(D)

        return R

class PrediNet(nn.Module):

    def __init__(self, image_shape, patch_size, d_model, n_heads, n_relations, n_classes):
        super(PrediNet, self).__init__()

        self.img_channels, self.img_width, self.img_height = image_shape
        self.patch_width, self.patch_height = patch_size
        self.n_classes = n_classes
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_relations = n_relations

        self.num_patches = (self.img_width // self.patch_width) * (self.img_height // self.patch_height)

        self.patch_dim = self.patch_width * self.patch_height * self.img_channels

        self.patch_embedder = create_patch_embedder(self.patch_height, self.patch_width, self.patch_dim, d_model)

        self.predinet = PrediNetModule(self.num_patches, d_model, n_relations, n_heads, add_temp_tag=True)

        predinet_out_dim = n_heads * (n_relations + 2) if self.predinet.add_temp_tag else n_heads * n_relations
        self.mlp_predictor = nn.Sequential(nn.Linear(predinet_out_dim, d_model), nn.ReLU(), nn.Linear(d_model, n_classes))

    def forward(self, x):
        x = self.patch_embedder(x)
        x = self.predinet(x)
        x = self.mlp_predictor(x)

        return x

class AbstractorModel(nn.Module):
    def __init__(self, image_shape, patch_size, d_model, abstractor_kwargs, num_classes):
        super(AbstractorModel, self).__init__()

        self.img_channels, self.img_width, self.img_height = image_shape
        self.patch_width, self.patch_height = patch_size
        self.num_classes = num_classes
        self.d_model = d_model

        self.num_patches = (self.img_width // self.patch_width) * (self.img_height // self.patch_height)

        self.patch_dim = self.patch_width * self.patch_height * self.img_channels

        self.patch_embedder = create_patch_embedder(self.patch_height, self.patch_width, self.patch_dim, d_model)

        self.abstractor = AbstractorModule(d_model=d_model, **abstractor_kwargs)

        self.mlp_predictor = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_classes))


    def forward(self, x):
        x = self.patch_embedder(x)
        x = self.abstractor(x)
        x = torch.mean(x, dim=1)
        x = self.mlp_predictor(x)

        return x