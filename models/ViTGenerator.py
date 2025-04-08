import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block
from einops.layers.torch import Rearrange


""" Here, we implement a ViT based image generator, that can be initialized with a pre-trained encoder.
This is the model that was used for one-shot and two-shot image prediction experiments in the thesis."""

class ViTEncoder(nn.Module):
    def __init__(self, 
                 image_size=64, 
                 num_channels=1, 
                 patch_size=4, 
                 emb_dim=384, 
                 num_layer=12, 
                 num_head=3, 
                 pretrained_model = None):
        
        super().__init__()
        self.patch_size = patch_size
        if pretrained_model is not None: 
            self.num_patches = pretrained_model.encoder.num_patches
            self.cls_token = pretrained_model.encoder.cls_token
            self.pos_embedding = pretrained_model.encoder.pos_embedding
            self.patchify = pretrained_model.encoder.patchify
            self.transformer = pretrained_model.encoder.transformer
            self.layer_norm = pretrained_model.encoder.layer_norm
        else:
            self.num_patches = (image_size // self.patch_size) ** 2
            self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
            self.patchify = nn.Conv2d(num_channels, emb_dim, patch_size, patch_size)
            self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
            self.layer_norm = nn.LayerNorm(emb_dim)
            
            self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        # patchify input image 
        patches = self.patchify(img)  
        patches = rearrange(patches, 'b c h w -> b (h w) c') # reshape from 2D conv output to [batch_size, num_patches, embed_dim]

        # add positional embeddings w/o cls_token
        patches = patches + self.pos_embedding 

        # concat cls_token
        cls_token_expanded = self.cls_token.expand(patches.size(0), -1, -1)  
        patches = torch.cat([cls_token_expanded, patches], dim=1) 
       
        # forward patches 
        features = self.transformer(patches)
        features = self.layer_norm(features)

        # output shape is [batch_size, num_patches + 1, embed_dim]
        return features


class ViTDecoder(nn.Module):
    def __init__(self, image_size=64, 
                 num_channels=1, 
                 patch_size=4, 
                 emb_dim=384, 
                 num_layer=4, 
                 num_head=3, 
                 mode='pairwise'):
        super().__init__()

        self.mode = mode
        self.mode_factor = 1 if self.mode=='pairwise' else 2
        self.num_patches = (image_size // patch_size) ** 2
        self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)]) 
        self.embedding2patch = nn.Linear(emb_dim, num_channels * patch_size ** 2)
        self.patch2img = Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size, c=num_channels)
        self.pos_embedding = nn.Parameter(torch.zeros(1, (self.num_patches + 1)*self.mode_factor, emb_dim))

        if self.mode == 'two_step':
                self.halve_embeddinggs = nn.Conv1d(in_channels=self.num_patches*2, out_channels=self.num_patches, kernel_size=1, stride=1)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features):
        # add positional embedding
        features = features + self.pos_embedding

        # forward through decoder transformer 
        features = self.transformer(features)
        
        # slice off the class token, shape becomes [batch_size, num_patches, embedding_dim]
        if self.mode == 'pairwise':
            features = features[:, 1:, :]  
        elif self.mode == 'two_step':
            features_one = features[:, 1:self.num_patches+1, :]  
            features_two = features[:, self.num_patches+2:, :]
            features = torch.cat(tensors=(features_one, features_two), dim=1)
            features = self.halve_embeddinggs(features)

        patches = self.embedding2patch(features)
        imgs = self.patch2img(patches)
        imgs = torch.sigmoid(imgs)  # Apply sigmoid to scale the output to [0, 1]

        return imgs

class ViTGenerator(nn.Module):
    def __init__(self, 
                 image_size=64, 
                 num_channels=1,
                 patch_size=4, 
                 emb_dim=384, 
                 encoder_layer=12, 
                 encoder_head=3, 
                 decoder_layer=4, 
                 decoder_head=3,
                 mode='pairwise',
                 pretrained_model=None):
        super().__init__()
        
        self.mode = mode

        self.encoder = ViTEncoder(image_size, num_channels, patch_size, emb_dim, encoder_layer, encoder_head, pretrained_model)
        self.decoder = ViTDecoder(image_size, num_channels, patch_size, emb_dim, decoder_layer, decoder_head, mode)

    def forward(self, img, additional_input=None):
        if self.mode == 'pairwise':
            features = self.encoder(img)
            generated_img = self.decoder(features)
        elif self.mode == 'two_step':
            features = self.encoder(img)

            if additional_input is None: 
                raise Exception(f"Expected 2 image inputs for mode {self.mode}, but got 1.")
            else:
                features_next_step = self.encoder(additional_input)
            features_concat = torch.cat(tensors=(features, features_next_step), dim=1)
            generated_img = self.decoder(features_concat)
        
        return generated_img
    