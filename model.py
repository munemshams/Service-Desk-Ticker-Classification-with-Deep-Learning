import torch
import torch.nn as nn

class TicketClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TicketClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x = self.conv(x)
        x = self.relu(x)

        x = self.pool(x).squeeze(2)

        x = self.fc(x)

        return x
