import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """
    A character-level language model using a vanilla Recurrent Neural Network (RNN).

    Theory:
    -------
    Recurrence:
    In an RNN, the output from the previous time step (hidden state) is fed back 
    as input to the current time step. This allows the network to maintain a 'memory' 
    of past inputs.
    
    Equation: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)

    Why RNNs struggle with long sequences:
    -------------------------------------
    The "Vanishing Gradient Problem". During backpropagation through time (BPTT), 
    gradients are calculated by multiplying weight matrices repeatedly. 
    If the weights are small (< 1), the gradient shrinks exponentially as it goes 
    back in time, causing the network to 'forget' long-range dependencies. 
    Conversely, if weights are large (> 1), gradients can explode.
    """
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # 1. Embedding Layer: Converts integer indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # 2. RNN Layer: The core recurrent component
        # batch_first=True means input shape is (batch, seq_len, input_size)
        self.rnn = nn.RNN(input_size=embed_size, hidden_size=hidden_size, batch_first=True)
        
        # 3. Output Head: Detailed logic to map hidden state to vocab distribution
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        """
        Args:
            x: Input tensor of shape (batch, seq_len)
            hidden: Initial hidden state of shape (1, batch, hidden_size)

        Returns:
            out: Output tensor of shape (batch, seq_len, vocab_size)
            hidden: Final hidden state
        """
        # Shape: (batch, seq_len) -> (batch, seq_len, embed_size)
        embeds = self.embedding(x)
        
        # Forward pass through RNN
        # out shape: (batch, seq_len, hidden_size)
        # hidden shape: (1, batch, hidden_size)
        out, hidden = self.rnn(embeds, hidden)
        
        # Reshape to pass through Linear layer efficiently: (batch * seq_len, hidden_size)
        out = out.reshape(-1, self.hidden_size)
        
        # Shape: (batch * seq_len, vocab_size)
        out = self.fc(out)
        
        return out, hidden

    def init_hidden(self, batch_size, device):
        """Initializes the hidden state with zeros."""
        return torch.zeros(1, batch_size, self.hidden_size).to(device)
