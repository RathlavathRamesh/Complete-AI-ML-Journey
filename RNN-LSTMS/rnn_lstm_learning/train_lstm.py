import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from models.lstm_model import SimpleLSTM
from models.common import Vocabulary
from utils.dataset import CharacterDataset
from utils.training_utils import train_one_epoch, print_sample_text

def main():
    # --- Configuration ---
    DATA_PATH = 'data/tiny_text.txt'
    BATCH_SIZE = 8
    SEQ_LEN = 30
    EMBED_SIZE = 32
    HIDDEN_SIZE = 64
    LR = 0.005
    EPOCHS = 200
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Training LSTM on {DEVICE}...")
    
    # --- Data Prep ---
    with open(DATA_PATH, 'r') as f:
        text = f.read()
        
    vocab = Vocabulary(text)
    dataset = CharacterDataset(text, vocab, SEQ_LEN)
    breakpoint()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- Model Setup ---
    model = SimpleLSTM(len(vocab), EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # --- Training Loop ---
    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(
            model, dataloader, loss_fn, optimizer, DEVICE
        )
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
            if (epoch+1) == EPOCHS:
                print_sample_text(model, vocab, "The cat", length=50, device=DEVICE)

    # --- Save Model ---
    torch.save(model.state_dict(), 'lstm_model.pth')
    print("LSTM Training Complete. Model saved to 'lstm_model.pth'.")

if __name__ == "__main__":
    main()
