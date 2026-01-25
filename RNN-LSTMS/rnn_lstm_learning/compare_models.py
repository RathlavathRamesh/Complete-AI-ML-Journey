import torch
import os
from models.rnn_model import SimpleRNN
from models.lstm_model import SimpleLSTM
from models.common import Vocabulary
from utils.training_utils import print_sample_text

def main():
    # --- Configuration ---
    DATA_PATH = 'data/tiny_text.txt'
    EMBED_SIZE = 32
    HIDDEN_SIZE = 64
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Comparing models on {DEVICE}...")
    
    # --- Data Prep (for Vocab) ---
    with open(DATA_PATH, 'r') as f:
        text = f.read()
    vocab = Vocabulary(text)
    
    # --- Load Models ---
    rnn_model = SimpleRNN(len(vocab), EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    lstm_model = SimpleLSTM(len(vocab), EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    
    if os.path.exists('rnn_model.pth'):
        rnn_model.load_state_dict(torch.load('rnn_model.pth', map_location=DEVICE))
        print("Loaded RNN model.")
    else:
        print("WARNING: rnn_model.pth not found. Please run train_rnn.py first.")
        
    if os.path.exists('lstm_model.pth'):
        lstm_model.load_state_dict(torch.load('lstm_model.pth', map_location=DEVICE))
        print("Loaded LSTM model.")
    else:
        print("WARNING: lstm_model.pth not found. Please run train_lstm.py first.")

    # --- Compare ---
    seed_text = "The cat"
    print("\n========================================")
    print("RNN GENERATION:")
    print("========================================")
    print_sample_text(rnn_model, vocab, seed_text, length=200, device=DEVICE)

    print("\n========================================")
    print("LSTM GENERATION:")
    print("========================================")
    print_sample_text(lstm_model, vocab, seed_text, length=200, device=DEVICE)

if __name__ == "__main__":
    main()
