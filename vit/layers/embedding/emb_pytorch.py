import torch
from torch import nn 
import argparse

class Embedding(nn.Module):
    def __init__(self, num_channels, patch_size, d_model):
        super(Embedding, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_channels = num_channels
        self.unf = nn.Unfold(kernel_size=patch_size, stride=patch_size) # stride = patch_size prevents overlapping patches
        self.proj = nn.Linear(num_channels * (patch_size ** 2), d_model)
        self.cls_emb = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, X):
        # X: (B, C, H, W)
        assert X.shape[2] % self.patch_size == 0 and X.shape[3] % self.patch_size == 0, "Height and width dimensions must be divisible by patch size"
        
        X = self.unf(X).permute(0, 2, 1) # (B, N, P*P*C)
        X = self.proj(X) # (B, N, D)

        return torch.cat((self.cls_emb, X), dim=1) # (B, N+1, D)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="__main__")
    parser.add_argument("--n_tests", type=int, default=1, help="number of tests to generate")
    parser.add_argument("--b_dim", type=int, default=1, help="batch dimension of the test input")
    parser.add_argument("--h_dim", type=int, default=4, help="height dimension of the test input")
    parser.add_argument("--w_dim", type=int, default=4, help="width dimension of the test input")
    parser.add_argument("--c_dim", type=int, default=2, help="channel dimension of the test input")
    parser.add_argument("--n_dim", type=int, default=2, help="patch dimension of the test input")
    parser.add_argument("--d_dim", type=int, default=2, help="embedding dimension of the test input")
    args = parser.parse_args()

    emb = Embedding(
        num_channels=args.c_dim,
        patch_size=args.n_dim,
        d_model=args.d_dim
    )
    for test_no in range(args.n_tests):
        input = torch.randn(size=(args.b_dim, args.h_dim, args.w_dim, args.c_dim))
        out = emb(input)

        print(f"TEST #{test_no} ------------")
        print(f"INPUT: {input}")
        print(f"OUTPUT: {out}")