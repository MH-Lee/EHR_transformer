import torch
import torch.nn as nn
import numpy as np
from src.models.layers import EncoderNew, TimeEncoder


class TransformerTimeGPT(nn.Module):
    def __init__(self, n_codes, batch_size, device, args, pretrained_emb=None):
        super(TransformerTimeGPT, self).__init__()
        self.device = device
        self.time_encoder = TimeEncoder(batch_size, device=self.device)
        self.feature_encoder = EncoderNew(vocab_size=n_codes, 
                                          max_seq_len=51,
                                          model_dim=args.model_dim,
                                          ffn_dim=args.ffn_dim,
                                          num_layers=args.num_layers, 
                                          device=self.device,
                                          pretrained_emb=pretrained_emb,
                                          diag_freeze=args.diag_freeze)

        self.self_layer = torch.nn.Linear(args.model_dim, 1)
        self.classify_layer = torch.nn.Linear(args.model_dim, args.num_classes)
        self.quiry_layer = torch.nn.Linear(args.model_dim, 64)
        self.quiry_weight_layer = torch.nn.Linear(args.model_dim, 2)
        self.relu = nn.ReLU(inplace=True)
        # dropout layer
        dropout_rate = args.dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill(mask, -np.inf), dim=1)
        return attention

    def forward(self, batch_data):
        # diagnosis_codes, lengths, seq_time_step, mask, mask_final, mask_code
        # diagnosis_codes: [batch_size, length, bag_len]
        # seq_time_step: [batch_size, length] the day times to the final visit
        # batch_labels: [batch_size, 100] 0 negative 1 positive for 100 disease
        diagnosis_codes = torch.tensor(batch_data['visit_seq'], dtype=torch.long, device=self.device)
        seq_time_step = batch_data['time_delta']
        lengths = batch_data['length'].to(self.device)
        mask_mult = torch.tensor(1-batch_data['seq_mask'], dtype=torch.bool, device=self.device).unsqueeze(2)
        mask_final = batch_data['seq_mask_final'].unsqueeze(2).to(self.device)
        mask_code = batch_data['seq_mask_code'].unsqueeze(3).to(self.device)

        features = self.feature_encoder(diagnosis_codes, mask_code, seq_time_step, lengths)
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.relu(self.quiry_layer(final_statues))

        self_weight = self.get_self_attention(features, mask_mult)
        time_weight = self.time_encoder(seq_time_step, quiryes, mask_mult)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        averaged_features = self.dropout(averaged_features)
        predictions = self.classify_layer(averaged_features).squeeze()
        return predictions, self_weight