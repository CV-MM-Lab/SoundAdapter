import torch
from torch import nn
from einops import rearrange, repeat


class Attention(nn.Module):
   def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
       super().__init__()
       inner_dim = dim_head *  heads   #64*8
       project_out = not (heads == 1 and dim_head == dim)

       self.heads = heads
       self.scale = dim_head ** -0.5
       self.attend = nn.Softmax(dim = -1)
       self.dropout = nn.Dropout(dropout)

       self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) #   512--->64*8*3
       self.to_out = nn.Sequential(
           nn.Linear(inner_dim, dim),
           nn.Dropout(dropout)
      ) if project_out else nn.Identity()

   def forward(self, x):
       qkv = self.to_qkv(x).chunk(3, dim = -1)
       q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
       dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
       attn = self.attend(dots)
       attn = self.dropout(attn)

       out = torch.matmul(attn, v)
       out = rearrange(out, 'b h n d -> b n (h d)')

       return self.to_out(out)

class FeedForward(nn.Module):
   def __init__(self, dim, hidden_dim, dropout = 0.):
       super().__init__()
       self.net = nn.Sequential(
           nn.Linear(dim, hidden_dim),
           nn.GELU(),
           nn.Dropout(dropout),
           nn.Linear(hidden_dim, dim),
           nn.Dropout(dropout)
      )
   def forward(self, x):
       return self.net(x)
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Transformer(nn.Module):
   def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
       super().__init__()
       self.layers = nn.ModuleList([])
       for _ in range(depth):
           self.layers.append(nn.ModuleList([
               PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
               PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
          ]))
   def forward(self, x):
       for attn, ff in self.layers:
           x = attn(x) + x
           x = ff(x) + x
       return x



class ViT(nn.Module):
   def __init__(self, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
       super().__init__()

       assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

       #self.pos_embedding = nn.Parameter(torch.randn(1, 78, dim))
       self.cls_token = nn.Parameter(torch.randn(1, 1, 512))

       self.dropout = nn.Dropout(emb_dropout)
       self.transformer1 = Transformer(dim, 80, heads, dim_head, mlp_dim, dropout)
       self.transformer2 = Transformer(dim, 80, heads, dim_head, mlp_dim, dropout)
       self.transformer3 = Transformer(dim, 80, heads, dim_head, mlp_dim, dropout)
       self.transformer4 = Transformer(dim, 80, heads, dim_head, mlp_dim, dropout)

       self.pool = pool
       self.to_latent = nn.Identity()

       self.mlp_head1 = nn.Sequential(
           nn.LayerNorm(dim),
           nn.Linear(dim, 512)
      )
       self.mlp_head2 = nn.Sequential(
           nn.LayerNorm(dim),
           nn.Linear(dim, 512)
       )
       self.mlp_head3 = nn.Sequential(
           nn.LayerNorm(dim),
           nn.Linear(dim, 512)
       )
       self.mlp_head4 = nn.Sequential(
           nn.LayerNorm(dim),
           nn.Linear(dim, 512)
       )

    #(1,1,512) ---> (1, 77, 768)
   def forward(self, x):
       b, n, _ = x.shape
       #1 77 768
       cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
       x = torch.cat((cls_tokens, x), dim=1)

       x = self.dropout(x)
       x1 = self.transformer1(x)
       x2 = self.transformer2(x1)
       x3 = self.transformer3(x2)
       x4 = self.transformer4(x3)

       x_head1 = x1[:, 0]
       x_head2 = x2[:, 0]
       x_head3 = x3[:, 0]
       x_head4 = x4[:, 0]
       x_head1 = self.to_latent(x_head1).unsqueeze(dim=1)
       x_head2 = self.to_latent(x_head2).unsqueeze(dim=1)
       x_head3 = self.to_latent(x_head3).unsqueeze(dim=1)
       x_head4 = self.to_latent(x_head4).unsqueeze(dim=1)

       x1 =  self.mlp_head1(x_head1)
       x2 = self.mlp_head2(x_head2)
       x3 = self.mlp_head3(x_head3)
       x4 = self.mlp_head4(x_head4)


       return x1+x2+x3+x4

if __name__ == '__main__':
    x = torch.randn(1,512)
    x = x.unsqueeze(0)
    x = x.repeat(500,1,1)
    vit = ViT(512,512,4,8,64)
    y = vit(x)
    print(y.shape)