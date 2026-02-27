
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from model import config
from model.modules import heads
from model.modules.bert_model import BertConfig, BertCrossLayer
from model.clip import build_model
from model.layers import Neck, Decoder, Projector
from model.fusion import Fusion
from model.dinov2.models.vision_transformer import vit_base,vit_large
cfg_ = config.config()
bert_config = BertConfig(
    vocab_size=cfg_["vocab_size"],
    hidden_size=cfg_["hidden_size"],
    num_hidden_layers=cfg_["num_layers"],
    num_attention_heads=cfg_["num_heads"],
    intermediate_size=cfg_["hidden_size"] * cfg_["mlp_ratio"],
    max_position_embeddings=cfg_["max_text_len"],
    hidden_dropout_prob=cfg_["drop_rate"],
    attention_probs_dropout_prob=cfg_["drop_rate"],
)
def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class MMFRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Text Encoder

        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.txt_backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.input_size, cfg.txtual_adapter_layer,cfg.txt_adapter_dim).float()
        self.fusion = Fusion(d_model=cfg.ladder_dim, nhead=cfg.nhead,dino_layers=cfg.dino_layers, output_dinov2=cfg.output_dinov2)
       # Fix Backbone
        for param_name, param in self.txt_backbone.named_parameters():
            if 'adapter' not in param_name :
                param.requires_grad = False

        state_dict = torch.load(cfg.dino_pretrain) 
        if cfg.dino_name=='dino-base':
            self.dinov2 = vit_base(
                patch_size=14,
                num_register_tokens=4,
                img_size=526,
                init_values=1.0,
                block_chunks=0,
                add_adapter_layer=cfg.visual_adapter_layer,
                visual_adapter_dim=cfg.visual_adapter_dim,                
            )
        else:
            self.dinov2=vit_large(
                patch_size=14,
                num_register_tokens=4,
                img_size=526,
                init_values=1.0,
                block_chunks=0,
                add_adapter_layer=cfg.visual_adapter_layer,
                visual_adapter_dim=cfg.visual_adapter_dim,                
            )
        self.dinov2.load_state_dict(state_dict, strict=False)

        for param_name, param in self.dinov2.named_parameters():
            if 'adapter' not in param_name:
                param.requires_grad = False


        self.cross_modal_text_manager_tower = nn.ModuleList(
            [heads.Manager(cfg_, 6, i) for i in range(cfg_['num_layers'])])
        self.cross_modal_image_manager_tower = nn.ModuleList(
            [heads.Manager(cfg_, 6, i) for i in range(cfg_['num_layers'])])
        # self.txt_emb = nn.Linear(512, 768)
        # self.txt_emb_ = nn.Linear(768, 512)
        # self.txt_emb = nn.Sequential(
        #     nn.Linear(512, 768),
        #     nn.GELU()
        #         )
        # self.txt_emb_ = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.GELU()
        #         )

        # 修改为视觉特征维度适配
        self.vis_emb = nn.Sequential(
             nn.Linear(768, 512),
             nn.GELU()
         )

        self.vis_emb_ = nn.Sequential(
             nn.Linear(512, 768),
             nn.GELU()
         )

        #self.vis_emb = nn.Linear(768, 512)
        #self.vis_emb_ = nn.Linear(512, 768)



        #self.extend_text_masks = torch.nn.Parameter(torch.ones((1, 1, 1, 17), device='cuda'), requires_grad=False)
        self.cross_modal_image_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(cfg_['num_layers'])])
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(bert_config) for _ in range(cfg_['num_layers'])])


        self.cross_modal_image_layers.apply(init_weights)
        self.cross_modal_text_layers.apply(init_weights)
        self.cross_modal_text_manager_tower.apply(init_weights)
        self.cross_modal_image_manager_tower.apply(init_weights)
        # Multi-Modal Decoder
        self.neck = Neck(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out, stride=cfg.stride)
        self.decoder = Decoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)

        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)


    def forward(self, img, word, mask=None, training=True):
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        vis, word, state= self.fusion(img, word, self.txt_backbone, self.dinov2)
        b, c, h, w = vis[0].shape
        # device = word.device
        image_embedss = torch.stack(vis, dim=1).view(vis[0].size(0), len(vis), vis[0].size(1), -1).transpose(2, 3)

        image_embedss = self.vis_emb(image_embedss)

        extend_image_masks = None
        #extend_text_masks = self.extend_text_masks
        extend_text_masks = None
        text_embedss = torch.stack(word, dim=1)
        # text_embedss = self.txt_emb(text_embedss)
        image_features = []
        for manager_layer_index in range(cfg_["num_layers"]):
            text_manager_tower = self.cross_modal_text_manager_tower[manager_layer_index]
            image_manager_tower = self.cross_modal_image_manager_tower[manager_layer_index]
            if manager_layer_index == 0:
                x1_ = text_manager_tower(text_embedss, 0, extend_text_masks, is_training=training)
                y1_ = image_manager_tower(image_embedss, 0, extend_image_masks, is_training=training)
            else:
                x1_ = text_manager_tower(text_embedss, x1, extend_text_masks, extra_query=y1,
                                         is_training=training)
                y1_ = image_manager_tower(image_embedss, y1, extend_image_masks, extra_query=x1,
                                          is_training=training)
            x1 = self.cross_modal_text_layers[manager_layer_index](x1_, y1_, extend_text_masks, extend_image_masks)[0]
            y1 = self.cross_modal_image_layers[manager_layer_index](y1_, x1_, extend_image_masks, extend_text_masks)[0]
            # image_features.append(y1.view(b, h, w, c).transpose(1, 3).transpose(2, 3))
            y1_recover = self.vis_emb_(y1)
            image_features.append(y1_recover.view(b, h, w, c).transpose(1, 3).transpose(2, 3))
        image_features = image_features[-3:]
        text_feats = x1
        #word = self.txt_emb_(text_feats)
        word = text_feats
        fq = self.neck(image_features, state)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, word, pad_mask)
        fq = fq.reshape(b, c, h, w)

        # b, 1, 104, 104
        pred = self.proj(fq, state)

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            loss = F.binary_cross_entropy_with_logits(pred, mask) 
            return pred.detach(), mask, loss
        else:
            return pred.detach()
