import torch.nn as nn
import torch


class CoherenceModel(nn.Module):
    def __init__(self, input_dim, proj_dim, hidden_dim, num_classes=2):
        super().__init__()
        self.log_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, proj_dim)
        )
        self.context_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        fused_feature_dim = proj_dim * 4
        
        self.scoring_head = nn.Sequential(
            nn.Linear(fused_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(fused_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, log_vec, ctx_vec):
        z_e = self.log_projector(log_vec)
        z_c = self.context_projector(ctx_vec)
        
        diff = torch.abs(z_e - z_c)
        prod = z_e * z_c
        fused_features = torch.cat([z_e, z_c, diff, prod], dim=-1)
        
        score = self.scoring_head(fused_features).squeeze(-1)
        logits = self.classifier_head(fused_features)
        
        return score, logits, z_e, z_c


