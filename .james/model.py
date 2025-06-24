import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

class ViT(nn.Module):
    def __init__(self, img_width, img_channels, patch_size, d_model, num_heads, num_layers, num_classes, ff_dim):
        super().__init__()

        self.patch_size = patch_size

        # given 7x7 flattened patch, map it into an embedding
        self.patch_embedding = nn.Linear(img_channels * patch_size * patch_size, d_model)

        # cls_token we are using we will be concatenating
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # (1, 4*4 + 1, 64)
        # + 1 because we add cls tokens
        self.position_embedding = nn.Parameter(
            torch.rand(1, (img_width // patch_size) * (img_width // patch_size) + 1, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # mapping 64 to 10 at the end
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        N, C, H, W = x.shape

        # we divide the image into 4 different 7x7 patches, and then flatten those patches
        # img shape will be 4*4 x 7*7
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(N, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(N, -1, C * self.patch_size * self.patch_size)

        # each 7*7 flatten patch will be embedded to 64 dim vector
        x = self.patch_embedding(x)

        # cls tokens concatenated after repeating it for the batch
        cls_tokens = self.cls_token.repeat(N, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)

        # learnable position embeddings added
        x = x + self.position_embedding

        # transformer takes 17x64 tensor, like it is a sequence with 17 words (17 because 4*4 + 1 from cls)
        x = self.transformer_encoder(x)

        # only taking the transformed output of the cls token
        x = x[:, 0]

        # mapping to number of classes
        x = self.fc(x)

        return x