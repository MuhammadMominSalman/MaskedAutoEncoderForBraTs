
import torch.nn as nn
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, frames=128, frame_patch_size=16, dim=100, img_size=128, patch_size=16,  channels=4):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        self.frames = frames
        self.image_size = img_size
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'
        self.patch_size = (patch_size, patch_size)
        self.frame_patch_size = frame_patch_size
        self.num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size
        self.rearrange = Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height,
                                   p2 = patch_width, pf = frame_patch_size)
        self.to_patch_embedding = nn.Linear(patch_dim, dim)

    def forward(self, x):
        x = self.rearrange(x)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        # B, C, H, W = x.shape
        # # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        return x