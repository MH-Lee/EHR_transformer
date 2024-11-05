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
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, max_len=50, 
                 num_classes=100, attn_dropout=0.1, dropout_rate=0.1, device='cpu', pretrained_emb=None):
        super(BERT, self).__init__()
        self.device = device
        if pretrained_emb is None:
            self.code_emb = Embedding(vocab_size, embed_dim, padding_idx=0)
        else:
            self.pretrained_emb = pretrained_emb[:, :embed_dim]
            self.diag_size = self.pretrained_emb.size(0)
            self.code_emb = Embedding(vocab_size, embed_dim, padding_idx=0)
            self.code_emb.weight.data[0:self.diag_size] = self.pretrained_emb
            
        self.positional_encoding = PositionalEncoding(embed_dim, dropout_rate, max_len)
        self.visit_segment_emb = Embedding(51, embed_dim, padding_idx=50)
        self.code_type_emb = Embedding(5, embed_dim, padding_idx=0)
        self.time_emb = TimeEncoder(embed_dim, device)
        
        self.act1 = nn.ReLU()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=attn_dropout,
                                                   dim_feedforward=hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classify_layer = nn.Linear(embed_dim, num_classes)
        self.act2 = nn.GELU()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, batch):
        code_embs = self.code_emb(batch['masked_visit_seq'].to(self.device))
        code_embs_pos = self.positional_encoding(code_embs.to(self.device))
        visit_seg = self.visit_segment_emb(batch['visit_segments'].to(self.device))
        code_types = self.code_type_emb(batch['code_types'].to(self.device))
        timedelta_emb = self.time_emb(batch['time_delta'].to(self.device),
                                      batch['seq_mask'].unsqueeze(2).to(self.device))
        input_embs = code_embs_pos + visit_seg + code_types + timedelta_emb
        
        encoder_output = self.encoder(input_embs, src_key_padding_mask=batch['seq_mask'].to(self.device))
        mlm_pos = batch['mask_pos'][:, :, None].expand(-1, -1, encoder_output.size(-1)).long()
        h_masked = torch.gather(encoder_output, 1, mlm_pos.to(self.device)) # masking position [batch_size, max_pred, d_model]
        h_masked = self.layer_norm(self.act2(self.fc1(h_masked))) # masking position [batch_size, max_pred, d_model]
        mlm_output = h_masked @ self.code_emb.weight.data.t() # masking position [batch_size, max_pred, n_diag]
        
        h_pooled = self.act1(self.fc2(encoder_output[:, 0, :])) # [batch_size, d_model]
        # import pdb; pdb.set_trace()
        logits_cls = self.classify_layer(h_pooled) # [batch_size, 100]
        
        if torch.isnan(mlm_output).any():
            import pdb; pdb.set_trace()
        return mlm_output, logits_cls