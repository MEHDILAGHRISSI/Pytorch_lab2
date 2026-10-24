"""Vision Transformer (ViT) from scratch sur MNIST."""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import DATA_DIR, MODELS_DIR, get_device, run_classification


class PatchEmbedding(nn.Module):
    """Découpe l'image en patchs et les projette en vecteurs de dimension `dim`."""

    def __init__(self, img_size=28, patch_size=7, in_ch=1, dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):                       # (B,1,28,28)
        x = self.proj(x)                        # (B,dim,4,4)
        return x.flatten(2).transpose(1, 2)     # (B,16,dim)


class Attention(nn.Module):
    """Multi-head self-attention écrite à la main."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # connexion résiduelle 1
        x = x + self.mlp(self.norm2(x))    # connexion résiduelle 2
        return x


class ViT(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_ch=1, num_classes=10,
                 dim=64, depth=4, heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_ch, dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.blocks(self.drop(x))
        return self.head(self.norm(x)[:, 0])   # classification via le token [CLS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = get_device()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = torchvision.datasets.MNIST(DATA_DIR, train=True, transform=transform, download=True)
    test_ds = torchvision.datasets.MNIST(DATA_DIR, train=False, transform=transform, download=True)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = ViT().to(device)
    print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    run_classification("vit_scratch", model, train_loader, test_loader, optimizer, device, args.epochs)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "vit_scratch.pth"))


if __name__ == "__main__":
    main()