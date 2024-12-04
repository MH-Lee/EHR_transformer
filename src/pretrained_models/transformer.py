import math
import torch
import torch.nn as nn
from .embed import PositionalEncoding, TimeEncoder
              

class Embedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx,
                                        max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                        sparse=sparse, _weight=_weight)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
                

class BERT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len=50, num_classes=100, 
                 attn_dropout=0.1, dropout_rate=0.1, device='cpu', pool_type='mean', pretrained_emb=None, mlm_loss_type='ce'):
        super(BERT, self).__init__()
        self.device = device
        self.pool_type = pool_type
        self.mlm_loss_type = mlm_loss_type
        if pretrained_emb is None:
            self.code_emb = Embedding(vocab_size, embed_dim, padding_idx=0)
        else:
            self.pretrained_emb = pretrained_emb[:, :embed_dim]
            self.diag_size = self.pretrained_emb.size(0) + 1
            self.code_emb = Embedding(vocab_size, embed_dim, padding_idx=0)
            self.code_emb.weight.data[1:self.diag_size] = self.pretrained_emb
            
        # self.positional_encoding = PositionalEncoding(embed_dim, dropout_rate, max_len)
        # self.visit_segment_emb = nn.Embedding(51, embed_dim, padding_idx=50)
        # nn.init.xavier_normal_(self.visit_segment_emb.weight.data)÷
        self.code_type_emb = nn.Embedding(5, embed_dim, padding_idx=0)
        nn.init.kaiming_normal_(self.code_type_emb.weight.data)        
        self.time_emb = TimeEncoder(embed_dim, device)
  
        self.act1 = nn.ReLU()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, 
                                                   nhead=num_heads, 
                                                   dropout=attn_dropout,
                                                   dim_feedforward=hidden_dim, 
                                                   batch_first=True,
                                                   activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder.apply(self.reset_parameters)
        
        if self.mlm_loss_type == 'ce':
            self.mlm_layer = nn.Linear(embed_dim, vocab_size)
        
        self.classify_layer = nn.Linear(embed_dim, num_classes)
        
        self.act2 = nn.GELU()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        
        if self.pool_type == 'concat':
            self.fc2 = nn.Linear(2*embed_dim, embed_dim)
        else:
            self.fc2 = nn.Linear(embed_dim, embed_dim)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def reset_parameters(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.kaiming_normal_(module.in_proj_weight)
            nn.init.kaiming_normal_(module.out_proj.weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)
    
    def forward(self, batch):
        code_embs = self.code_emb(batch['masked_visit_seq'].to(self.device))
        # code_embs_pos = self.positional_encoding(code_embs.to(self.device))
        # visit_seg = self.visit_segment_emb(batch['visit_segments'].to(self.device))
        code_types = self.code_type_emb(batch['code_types'].to(self.device))
        timedelta_emb = self.time_emb(batch['time_delta'].to(self.device),
                                      batch['seq_mask'].unsqueeze(2).to(self.device))
        input_embs = code_embs + code_types + timedelta_emb
        
        encoder_output = self.encoder(input_embs, src_key_padding_mask=batch['seq_mask'].to(self.device))
        mlm_pos = batch['mask_pos'][:, :, None].expand(-1, -1, encoder_output.size(-1)).long()
        h_masked = torch.gather(encoder_output, 1, mlm_pos.to(self.device)) # masking position [batch_size, max_pred, d_model]
        h_masked = self.layer_norm(self.act2(self.fc1(h_masked))) # masking position [batch_size, max_pred, d_model]
        
        if self.mlm_loss_type == 'ce':
            mlm_output = self.mlm_layer(h_masked)
            # mlm_output = h_masked @ self.code_emb.weight.data.t() # masking position [batch_size, max_pred, vocab_size]
        elif self.mlm_loss_type == 'mse':
            mlm_output = h_masked
        else:
            raise ValueError("Invalid mlm loss type selected %s" % self.mlm_loss_type)
        
        if self.pool_type == 'mean':
            # import pdb; pdb.set_trace()
            non_zero = ((batch['code_types'] != 4) & (batch['code_types'] != 0)).float().unsqueeze(2).to(self.device)
            mean_pool = (encoder_output * non_zero).sum(1) / non_zero.sum(1)
            h_pooled = self.act1(self.fc2(mean_pool)) # [batch_size, d_model]
        elif self.pool_type == 'concat':
            non_zero = ((batch['code_types'] != 4) & (batch['code_types'] != 0)).float().unsqueeze(2).to(self.device)
            mean_pool = (encoder_output * non_zero).sum(1) / non_zero.sum(1)
            cls_pool = encoder_output[:, 0, :]
            concat_pool = torch.cat([cls_pool, mean_pool], dim=1)
            h_pooled = self.act1(self.fc2(concat_pool)) # [batch_size, d_model]
        elif self.pool_type == 'cls':
            h_pooled = self.act1(self.fc2(encoder_output[:, 0, :])) # [batch_size, d_model]
        else:
            raise ValueError("Invalid pool type selected %s" % self.pool_type)
        
        # import pdb; pdb.set_trace()
        logits_cls = self.classify_layer(h_pooled) # [batch_size, 100]
        
        if torch.isnan(mlm_output).any():
            import pdb; pdb.set_trace()
            
        return mlm_output, logits_cls