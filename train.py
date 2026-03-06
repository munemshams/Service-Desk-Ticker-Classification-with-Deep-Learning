import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from model import TicketClassifier

MAX_LEN = 50
EMBED_DIM = 64
EPOCHS = 3
LR = 0.05

with open("words.json") as f:
    words = json.load(f)

with open("text.json") as f:
    text = json.load(f)

labels = np.load("labels.npy")

word2idx = {w: i+1 for i, w in enumerate(words)}
vocab_size = len(word2idx) + 1

def encode(sentence):

    encoded = [word2idx.get(w, 0) for w in sentence]

    if len(encoded) < MAX_LEN:
        encoded += [0]*(MAX_LEN-len(encoded))
    else:
        encoded = encoded[:MAX_LEN]

    return encoded

X = np.array([encode(s) for s in text])
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)

model = TicketClassifier(vocab_size, EMBED_DIM, len(set(labels)))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model.pth")

print("Model saved as model.pth")
