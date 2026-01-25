import torch
import torch.nn as nn

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    
    Args:
        model: The RNN or LSTM model.
        dataloader: DataLoader providing (x, y) batches.
        criterion: Loss function (usually CrossEntropyLoss).
        optimizer: Optimizer (usually Adam).
        device: 'cuda' or 'cpu'.
        
    Returns:
        avg_loss: Float, average loss for the epoch.
    """
    model.train()
    total_loss = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)
        
        # Initialize hidden state for the current batch (stateless training)
        hidden = model.init_hidden(batch_size, device)
        
        # Forward pass
        # out shape: (batch * seq_len, vocab_size)
        # hidden: final hidden state (ignored for stateless training)
        out, hidden = model(x, hidden)
        
        # Reshape y to match out: (batch * seq_len)
        y = y.view(-1)
        
        loss = criterion(out, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients (common in RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def print_sample_text(model, vocab, seed_text, length=100, device='cpu'):
    """
    Generates text using the trained model.
    """
    model.eval()
    
    # 1. Prepare initial hidden state
    hidden = model.init_hidden(1, device)
    
    # 2. Prime the model with the seed text
    indices = vocab.encode(seed_text)
    input_seq = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device) # (1, seq_len)
    
    generated_text = seed_text
    
    with torch.no_grad():
        # Pass seed text to update hidden state
        # We only care about the final hidden state after the seed
        _, hidden = model(input_seq, hidden)
        
        # The last character of the seed is the first input for generation
        current_input = input_seq[:, -1].unsqueeze(1) # (1, 1)
        
        for _ in range(length):
            out, hidden = model(current_input, hidden)
            
            # out shape: (1 * 1, vocab_size) -> (vocab_size)
            logits = out.squeeze()
            
            # Greedy decoding: pick the character with highest probability
            predicted_ix = torch.argmax(logits).item()
            predicted_char = vocab.ix_to_char[predicted_ix]
            
            generated_text += predicted_char
            
            # Feed the predicted character as input for the next step
            current_input = torch.tensor([[predicted_ix]], dtype=torch.long).to(device)
            
    print(f"--- Generated Text (Seed: '{seed_text}') ---")
    print(generated_text)
    print("----------------------------------------------\n")
    return generated_text
