""" Vision Transformer (ViT) in PyTorch
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from VIT.layers.patch_embd import LinearPatchEmbed, ConvPatchEmbed, PositionEmbed2D
from VIT.layers.mlp import Mlp
from VIT.layers.drop import DropPath

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PromptsPool(nn.Module):
    def __init__(self, in_features, num_tokens=None, num_heads = 12):
        super().__init__()
        if num_tokens is None:
            num_tokens = 196
        self.in_features = in_features
        self.num_tokens = num_tokens
        self.scale = in_features ** -0.5
        self.num_heads = num_heads
        #self.tokens_no_decay = nn.Parameter(torch.randn(2, self.num_heads, num_tokens, in_features//self.num_heads))
        self.tokens = nn.Parameter(torch.randn(2, self.num_heads, num_tokens, in_features//self.num_heads))
        self.fc = nn.Linear(in_features, in_features)
        #self.proj = nn.Linear(in_features, in_features)

    def forward(self, x, y):
        #x = self.red(x)
        #skip = x
        B, N, C = x.shape
        #q = self.tokens.repeat(B, 1, 1, 1)
        q = torch.index_select(self.tokens, 0, y.int())
        k = self.fc(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        #k, v = kv.unbind(0) #B, 12, 196, C//12
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) #B, 12, 32, 196
        # B, 12, 32, C//12
        #print(attn.shape, x.shape)
        x = (attn @ v).transpose(1, 2).reshape(B, self.num_tokens, C)
        #x = self.proj(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, post_scale=1.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.adjust_scale = post_scale
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        #where to norm
        if self.adjust_scale == 0: # pre-norm
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.adjust_scale == 1: # post-norm
            x = x + self.drop_path(self.norm1(self.attn(x)))
            x = x + self.drop_path(self.norm2(self.mlp(x)))
        else: # deep-norm
            x = self.norm1(x * self.adjust_scale + self.drop_path(self.attn(x)))
            x = self.norm2(x * self.adjust_scale + self.drop_path(self.mlp(x)))
        
        return x

class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=512, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 embed_layer=LinearPatchEmbed, pos_embed="cosine", norm_layer=nn.LayerNorm, act_layer=nn.GELU, pool='mean',
                 classification=False, fp16=False, ntype="deepnorm", n_prompt_layer=0):
                 #):
        super().__init__()
        self.fp16 = fp16
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 0 #0 #1  
        self.classification = classification 
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
    
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        
        if pos_embed == "cosine":
            self.pos_embed = PositionEmbed2D(self.patch_embed.grid_size, embed_dim, self.num_tokens)()
            # PositionEmbed(num_patches, embed_dim, self.num_tokens)()
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
       
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        if ntype == "deepnorm":
            self.post_scale = (2*depth)**(0.25)
            self.init_scale = (8*depth)**(-0.25)
        elif ntype == "prenorm":
            self.post_scale = 0
            self.init_scale = 1
        elif ntype == "postnorm":
            self.post_scale = 1
            self.init_scale = 1
        else:
            raise ValueError
        
        self.depth = depth
        self.n_prompt_layer = n_prompt_layer

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                post_scale=self.post_scale)
            for i in range(depth)])
        
        self.pool = pool
        if n_prompt_layer>0:
            self.nt = 32
            self.prompts = nn.Parameter(torch.randn(n_prompt_layer, 2, self.nt, embed_dim))
            self.prompts_pool = PromptsPool(in_features=embed_dim, num_tokens=self.nt, num_heads=num_heads)
            
        if self.classification:
            self.class_head = nn.Sequential(
                                        nn.Linear(self.num_features, self.num_classes),
                                        nn.BatchNorm1d(self.num_classes, affine=False),) 
        
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, module):
        """ ViT weight initialization
        """
        if isinstance(module, nn.Linear):
            if module.out_features == self.num_classes:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                #trunc_normal_(module.weight, std=.02)
                nn.init.xavier_uniform_(module.weight, gain=self.init_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight, gain=self.init_scale)
            nn.init.xavier_uniform_(module.out_proj.weight, gain=self.init_scale)

        elif isinstance(module, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}
    
    def forward_features(self, x, y):
        """Return the layernormalization features
        """
        with torch.cuda.amp.autocast(self.fp16):
            x = self.patch_embed(x)
            x = self.pos_drop(x + self.pos_embed)

            if self.n_prompt_layer > 0:
                prompt_tokens = torch.index_select(self.prompts, 1, y.int())

            for i in range(self.depth):
                if i < self.n_prompt_layer: # add prompt before transformer blocks
                    x = torch.cat((prompt_tokens[i], x), dim=1)
                x = self.blocks[i](x)
                if i < self.n_prompt_layer:
                    x = x[:,self.nt:,:] # remove previous prompt

            if self.n_prompt_layer>0:
                 x = self.prompts_pool(x, y)

        return x.float() if self.fp16 else x #(x.float(),None) if self.fp16 else (x,None)

    def forward(self, x, y=0):
        x = self.forward_features(x, y)
            
        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "cls":
            x = x[:, 0]  # cls token
        elif self.pool == "adj":
            pool_weight = torch.nn.functional.softmax(x, dim=1)
            x = (x * pool_weight).sum(dim=1)
        else:
            raise ValueError("pool must be 'cls' or 'mean' or 'adj' ")
        
        assert x.shape[1] == self.num_features, "outputs must be same with the features"
     
        if self.classification:
            x = self.class_head(x)

        return x


class MAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder_dim = 516,
        masking_ratio = 0.75,
        fp16=False, ntype="deepnorm"
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio
        self.fp16 = fp16

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        encoder_dim = encoder.num_features
        pixel_values_per_patch = self.encoder.patch_embed.to_token.weight.shape[-1]

        # decoder parameters
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim))
        self.decoder = VisionTransformer(img_size=112, 
                                in_chans=3,
                                num_classes=512,
                                patch_size=8, 
                                embed_dim=decoder_dim, 
                                depth=8, 
                                num_heads=12,
                                mlp_ratio=3., 
                                qkv_bias=True,
                                embed_layer=LinearPatchEmbed, 
                                pos_embed="cosine", 
                                norm_layer=nn.LayerNorm, 
                                act_layer=nn.GELU, 
                                pool='mean',
                                ntype=ntype,
                                fp16=fp16
                                )
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img):
        device = img.device

        # get patches

        patches = self.encoder.patch_embed.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.encoder.patch_embed.norm(self.encoder.patch_embed.to_token(patches))
        tokens = tokens + self.encoder.pos_embed

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]#torch.index_select(tokens, 1, unmasked_indices) #

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices] #torch.index_select(patches, 1, masked_indices) #

        # attend with vision transformer
        if self.fp16:
            with torch.cuda.amp.autocast():
                encoded_tokens = self.encoder.blocks(tokens)#self.encoder.norm(self.encoder.blocks(tokens))

                # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

                decoder_tokens = self.enc_to_dec(encoded_tokens)

                # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

                mask_tokens = self.mask_token.repeat(batch, num_masked, 1)
                dec_pos_emb = self.decoder.pos_embed.repeat(batch, 1, 1)
                mask_tokens = mask_tokens + dec_pos_emb[batch_range, masked_indices]#torch.index_select(self.decoder.pos_embed, 1, masked_indices) #
                
                # concat the masked tokens to the decoder tokens and attend with decoder

                decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim = 1)
                decoded_tokens = self.decoder.blocks(decoder_tokens)#self.decoder.norm(self.decoder.blocks(decoder_tokens))

                # splice out the mask tokens and project to pixel values

                mask_tokens = decoded_tokens[:, :num_masked]
                pred_pixel_values = self.to_pixels(mask_tokens)

                # calculate reconstruction loss

                recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
                pred_pixel_values = pred_pixel_values.float()
        else:
            encoded_tokens = self.encoder.blocks(tokens)#self.encoder.norm(self.encoder.blocks(tokens))

            # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

            decoder_tokens = self.enc_to_dec(encoded_tokens)

            # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

            mask_tokens = self.mask_token.repeat(batch, num_masked, 1)
            dec_pos_emb = self.decoder.pos_embed.repeat(batch, 1, 1)
            mask_tokens = mask_tokens + dec_pos_emb[batch_range, masked_indices]#torch.index_select(self.decoder.pos_embed, 1, masked_indices) #
            
            # concat the masked tokens to the decoder tokens and attend with decoder

            decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim = 1)
            decoded_tokens = self.decoder.blocks(decoder_tokens)#self.decoder.norm(self.decoder.blocks(decoder_tokens))

            # splice out the mask tokens and project to pixel values

            mask_tokens = decoded_tokens[:, :num_masked]
            pred_pixel_values = self.to_pixels(mask_tokens)

            # calculate reconstruction loss

            recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
            
        return recon_loss, pred_pixel_values, masked_indices


# =====================================================================================================

def facet_base(**kwargs):
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = VisionTransformer(img_size=112, 
                              in_chans=3,
                              num_classes=512,
                              mlp_ratio=4., 
                              qkv_bias=True,
                              embed_layer=LinearPatchEmbed, 
                              pos_embed="learn", 
                              norm_layer=nn.LayerNorm, 
                              act_layer=nn.GELU, 
                              pool='mean',
                              **model_kwargs
                              )
    return model
    