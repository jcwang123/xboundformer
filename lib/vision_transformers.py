import torch
from torch import nn

from lib.transformer import BoundaryAwareTransformer
from lib.Position_embedding import PositionEmbeddingLearned


class in_scale_transformer(nn.Module):
    def __init__(self,
                 point_pred_layers=1,
                 num_queries=1,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=nn.LeakyReLU,
                 normalize_before=False,
                 return_intermediate_dec=False,
                 BAG_type='2D',
                 Atrous=False):

        super().__init__()

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.pos_embed = PositionEmbeddingLearned(d_model // 2)
        self.num_queries = num_queries

        self.transformer = BoundaryAwareTransformer(
            point_pred_layers=point_pred_layers,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
            return_intermediate_dec=return_intermediate_dec,
            BAG_type=BAG_type,
            Atrous=Atrous)

    def forward(self, x):

        pos_embed = self.pos_embed(x).to(x.dtype)

        latent_tensor, features_encoded, point_maps = self.transformer(
            x, None, self.query_embed.weight, pos_embed)

        return latent_tensor, features_encoded, point_maps


# def detr_Transformer(pretrained=False, **kwargs):

#     transformer = DETR_Transformer(num_encoder_layers=6,
#                                    num_decoder_layers=6,
#                                    d_model=256,
#                                    nhead=8)

#     if pretrained:
#         print("Loaded DETR Pretrained Parameters From ImageNet...")
#         ckpt = torch.load(
#             '/home/chenfei/my_codes/TransformerCode-master/Ours/pretrained/detr-r50-e632da11.pth'
#         )
#         state_dict = ckpt['model']

#         transformer.query_embed = nn.Embedding(100, 256)
#         unParalled_state_dict = {}
#         for key in state_dict.keys():
#             if key.startswith("transformer"):
#                 unParalled_state_dict[key] = state_dict[key]
#             #elif key.startswith("query_embed"):
#             #    unParalled_state_dict[key] = state_dict[key]
#         ### Without positional embedding for detr and mismatch for query_embed
#         transformer.load_state_dict(unParalled_state_dict, strict=False)
#     return transformer
