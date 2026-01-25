import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from models.rnn_model import SimpleRNN
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
    EPOCHS = 200 # Short text needs many epochs to memorize
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Training RNN on {DEVICE}...")
    
    # --- Data Prep ---
    with open(DATA_PATH, 'r') as f:
        text = f.read()
        
    vocab = Vocabulary(text)
    dataset = CharacterDataset(text, vocab, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Vocabulary Size: {len(vocab)}")
    print(f"Total batches: {len(dataloader)}")
    
    # --- Model Setup ---
    model = SimpleRNN(len(vocab), EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # --- Training Loop ---
    for epoch in range(EPOCHS):
        avg_loss = train_one_epoch(
            model, dataloader, loss_fn, optimizer, DEVICE
        )
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
            # Quick check
            if (epoch+1) == EPOCHS:
                print_sample_text(model, vocab, "The cat", length=50, device=DEVICE)

    # --- Save Model ---
    torch.save(model.state_dict(), 'rnn_model.pth')
    print("RNN Training Complete. Model saved to 'rnn_model.pth'.")

if __name__ == "__main__":
    main()
