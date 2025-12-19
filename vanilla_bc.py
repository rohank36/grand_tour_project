import torch
import torch.nn as nn

class VanillaBC(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(VanillaBC, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
            # Add nn.Tanh() here if actions are normalized to [-1, 1]
        )

    def forward(self, obs_history):
        # obs_history: [Batch, History_Window * State_Dim]
        return self.net(obs_history)

# Loss and Optimizer
model = VanillaBC(input_dim=obs_dim * history_len, action_dim=12)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)