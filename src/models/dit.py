import torch 
import torch.nn as nn
import torch.nn.functional as F
# dit model:
# 本质：vit模型改动而成，加入条件调制
# 结构：
# 输入(B,C,H,W)->PatchEmbed(conv2d+reshape)->tokens->pos_embedding->12*DiTBlock->u头v头
# t、r、w分别正弦编码并经过独立MLP，y通过Embedding查表，结果相加
def _modulate(x, scale, shift):
    scale=scale.unsqueeze(1)
    shift=shift.unsqueeze(1)
    return x * (1 + scale) + shift
class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps=None,use_weight=True):
        super().__init__()
        self.eps=eps#极小数，避免除以零
        self.register_parameter('weight', nn.Parameter(torch.ones(dim)) if use_weight else None)
    def _norm(self,x:torch.Tensor):
        eps = torch.finfo(x.dtype).eps if self.eps is None else self.eps
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps)
    def forward(self,x:torch.Tensor):
        out = self._norm(x)
        return out if self.weight is None else self.weight * out
class MHAttention(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_head):
        super().__init__()
        assert hidden_dim % num_head == 0
        self.q_proj=nn.Linear(input_dim,hidden_dim)
        self.k_proj=nn.Linear(input_dim,hidden_dim)
        self.v_proj=nn.Linear(input_dim,hidden_dim)
        
        self.out_proj=nn.Linear(hidden_dim,input_dim)
        self.head_dim=int(hidden_dim // num_head)
        self.q_norm=RMSNorm(self.head_dim) 
        self.k_norm=RMSNorm(self.head_dim)
        self.num_head=num_head
        self.scale_factor=self.head_dim**(-0.5)
    def forward(self,x):
        B,N,C=x.shape
        Q=self.q_proj(x).reshape(B,N,-1,self.head_dim).permute(0,2,1,3)
        Q=self.q_norm(Q)
        K=self.k_proj(x).reshape(B,N,-1,self.head_dim).permute(0,2,1,3)
        K=self.k_norm(K)
        V=self.v_proj(x)
        V=V.reshape(B,N,-1,self.head_dim).permute(0,2,1,3)
        logit=Q @ K.transpose(-1,-2) * self.scale_factor#(B,num_head,N,N)
        A=torch.softmax(logit,dim=-1)
        O=(A @ V).permute(0,2,1,3).reshape(B,N,-1)
        return self.out_proj(O)
class PatchEmb(nn.Module):
    """
    kernel_size = stride = P的conv2d,相当于下采样为(B,C,H//P,W//P)
    然后flatten成(B,C,N)在reshape输出
    """
    def __init__(self,patch_size,in_channel,hidden_dim,image_shape=(32,32)):
        super().__init__()
        assert image_shape[0] == image_shape[1]
        assert image_shape[0] % patch_size == 0 and hidden_dim % 4 == 0
        self.image_shape = tuple(image_shape)
        self.proj=nn.Conv2d(in_channel,hidden_dim,kernel_size=patch_size,stride=patch_size)
        grid_size=image_shape[0]//patch_size
        pos=self._build_pos_embed(grid_size,hidden_dim)
        self.pos_embed=nn.Parameter(pos.unsqueeze(0))
    @staticmethod
    def _build_pos_embed(grid_size,dim):
        # col、row两个坐标分别正弦位置编码，拼接嵌入
        # col的维度为dim//2,sin、cos分别占dim//4
        coords=torch.arange(grid_size)
        row,col = torch.meshgrid(coords,coords,indexing="ij")
        # 构建坐标系：row=[[1,..,1],..,[N,..,N]],col=[[1,...,N],...,[1,...,N]]
        freq = 10000.0 ** (
            -torch.arange(dim // 4, dtype=torch.float64) / (dim // 4)
        )#sin|cos(pos/(10000**(i/dim)))
        def encode(coord):
            angles=coord.reshape(-1,1)*freq
            return torch.cat([angles.sin(),angles.cos()],dim=-1)
        return torch.cat([encode(col),encode(row)],dim=-1).float()
    def forward(self,x):
        assert x.shape[-2:] == self.image_shape
        x = self.proj(x)
        x=x.flatten(2).transpose(1,2)
        return x + self.pos_embed
class ScalerEmd(nn.Module):
    def __init__(self,hidden_dim,embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj=nn.Sequential(
            nn.Linear(embed_dim,hidden_dim), nn.SiLU(), nn.Linear(hidden_dim,hidden_dim)
        )
    def _sinusoidal_emb(self,input):
        # 将标量进行正弦编码
        input = input.float() * 1000
        half = self.embed_dim // 2
        freqs = 10000.0 ** (-torch.arange(half, device=input.device, dtype=torch.float32) / half)
        args = input[:, None] * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return F.pad(emb, (0, self.embed_dim % 2))
    def forward(self,x):
        # x:(B,) float
        emb=self._sinusoidal_emb(x)
        return self.proj(emb)
class LabelEmb(nn.Module):
    def __init__(self,hidden_dim,num_class):
        super().__init__()
        self.embedding=nn.Embedding(num_class+1,hidden_dim)
    def forward(self,x):
        # x: (B,), torch.long
        return self.embedding(x)
class DiTBlock(nn.Module):
    def __init__(self,hidden_dim,num_head,mlp_scale=4.0):
        super().__init__()
        mlp_expand_dim=int(mlp_scale*hidden_dim)
        self.mlp=nn.Sequential(
            nn.Linear(hidden_dim,mlp_expand_dim),nn.GELU(approximate='tanh'),
            nn.Linear(mlp_expand_dim,hidden_dim)
        )
        self.norm_mlp=RMSNorm(hidden_dim,use_weight=False)
        self.attn=MHAttention(input_dim=hidden_dim,hidden_dim=hidden_dim,num_head=num_head)
        self.norm_attn=RMSNorm(hidden_dim,use_weight=False)
        self.adaLN_modulation=nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim)
        )#提供attn路和mlp路的scale、shift、gate用于调制，一共6个向量
    
    def forward(self,x,c):
        # c为条件
        attn_shift,attn_scale,attn_gate,mlp_shift,mlp_scale,mlp_gate=(
            self.adaLN_modulation(c).chunk(6,dim=-1)
        )
        x=x+attn_gate.unsqueeze(1)*self.attn(_modulate(self.norm_attn(x),attn_scale,attn_shift))
        x=x+mlp_gate.unsqueeze(1)*self.mlp(_modulate(self.norm_mlp(x),mlp_scale,mlp_shift))
        return x
class UnPatchLayer(nn.Module):
    def __init__(self,hidden_dim,out_channel,patch_size,image_shape=(32,32)):
        super().__init__()
        self.patch_size=patch_size
        self.image_shape=image_shape
        self.hidden_size=(image_shape[0]//patch_size , image_shape[1]//patch_size) 
        self.out_channel=out_channel
        self.norm=RMSNorm(hidden_dim,use_weight=False)
        self.proj=nn.Linear(hidden_dim,patch_size**2 * out_channel)
        self.adaLN_modulation=nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
    def forward(self,x,c):
        # (B,H/P * W/P ,C')->(B,H/P * W/P,3*P*P)->(B,H/P,W/P,P,P,3)->(B,H,W,3)
        batch_size=x.shape[0]
        shift,scale=self.adaLN_modulation(c).chunk(2,dim=-1)
        x=_modulate(self.norm(x),scale,shift)
        patch_expand=self.proj(x)
        unpatched=patch_expand.reshape(batch_size,self.hidden_size[0],self.hidden_size[1],self.patch_size,self.patch_size,self.out_channel)
        out=unpatched.permute(0,5,1,3,2,4).reshape(batch_size,self.out_channel,self.image_shape[0],self.image_shape[1])
        return out 
class DiT_meanflow(nn.Module):
    def __init__(self,
                 image_size=(32,32),
                 patch_size=2,
                 in_channel=3,
                 hidden_dim=512,
                 depth=12,
                 num_head=8,
                 mlp_scale=4.0,
                 num_classes=10,
                 max_cfg=7.5):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.max_cfg = max_cfg
        self.num_classes = num_classes
        self.x_embedder = PatchEmb(patch_size, in_channel, hidden_dim, image_size)
        self.t_embedder = ScalerEmd(hidden_dim)
        self.r_embedder = ScalerEmd(hidden_dim)
        self.w_embedder = ScalerEmd(hidden_dim)
        self.y_embedder = LabelEmb(hidden_dim, num_classes) if num_classes is not None else None
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_head, mlp_scale) for _ in range(depth)
        ])
        self.final_layer_u = UnPatchLayer(hidden_dim, in_channel, patch_size, image_size)
        self.final_layer_v = UnPatchLayer(hidden_dim, in_channel, patch_size, image_size)
        self.initialize_weights()

    def initialize_weights(self):
        def init_linear(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(init_linear)
        # 原版合并QKV使用[3D, D]的Xavier初始化，再分给三个投影。
        for block in self.blocks:
            attn = block.attn
            qkv = attn.q_proj.weight.new_empty(3 * attn.q_proj.out_features, attn.q_proj.in_features)
            nn.init.xavier_uniform_(qkv)
            with torch.no_grad():
                for layer, weight in zip((attn.q_proj, attn.k_proj, attn.v_proj), qkv.chunk(3)):
                    layer.weight.copy_(weight)
        proj = self.x_embedder.proj
        nn.init.xavier_uniform_(proj.weight.view(proj.weight.shape[0], -1))
        nn.init.zeros_(proj.bias)
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)
        for embedder in (self.t_embedder, self.r_embedder, self.w_embedder):
            nn.init.normal_(embedder.proj[0].weight, std=0.02)
            nn.init.normal_(embedder.proj[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        for head in (self.final_layer_u, self.final_layer_v):
            nn.init.zeros_(head.adaLN_modulation[-1].weight)
            nn.init.zeros_(head.adaLN_modulation[-1].bias)
            nn.init.zeros_(head.proj.weight)
            nn.init.zeros_(head.proj.bias)

    def forward(self, x, t, r, y=None, w=None):
        x = self.x_embedder(x)  # PatchEmb内已经添加位置编码
        if w is None:
            w = torch.ones_like(t)
        c = self.t_embedder(t) + self.r_embedder(r) 
        if w is not None:
            c=c+self.w_embedder(w / self.max_cfg)#scaler里面需要归一化到0，1.这里遵从原版做法处max_cfg进行归一化
        if y is not None:
            c = c + self.y_embedder(y)
        for block in self.blocks:
            x = block(x, c)
        u = self.final_layer_u(x, c)
        v= self.final_layer_v(x, c)
        return u, v
