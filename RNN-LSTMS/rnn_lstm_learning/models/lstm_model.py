import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    """
    A character-level language model using a Long Short-Term Memory (LSTM) network.

    Why LSTM is better than RNN:
    ----------------------------
    LSTMs are designed to mitigate the vanishing gradient problem. They introduce
    a more complex internal structure called a "cell" with three gates:
    1. Forget Gate: What information to discard from the cell state.
    2. Input Gate: What new information to store in the cell state.
    3. Output Gate: What information to output based on the cell state.

    Cell State vs. Hidden State:
    ----------------------------
    - Cell State (c_t): The "long-term memory" highway. It runs straight down the entire chain 
      with only minor linear interactions, making it easy for information to flow unchanged.
    - Hidden State (h_t): The "short-term memory" or the output of the LSTM cell at the 
      current step, used for predictions and passed to the next step.
    """
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # 2. LSTM Layer
        # Unlike RNN, LSTM returns (hidden_state, cell_state) tuple
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True)
        
        # 3. Output Head
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        """
        Args:
            x: Input tensor of shape (batch, seq_len)
            hidden: Tuple (h_0, c_0), each shape (1, batch, hidden_size)

        Returns:
            out: Output tensor of shape (batch, seq_len, vocab_size)
            hidden: Tuple (h_n, c_n) - final states
        """
        # Shape: (batch, seq_len, embed_size)
        embeds = self.embedding(x)
        
        # Forward pass through LSTM
        # out shape: (batch, seq_len, hidden_size)
        # hidden is a tuple: (h_n, c_n)
        out, hidden = self.lstm(embeds, hidden)
        
        # Reshape for Linear layer: (batch * seq_len, hidden_size)
        out = out.reshape(-1, self.hidden_size)
        
        # Shape: (batch * seq_len, vocab_size)
        out = self.fc(out)
        
        return out, hidden

    def init_hidden(self, batch_size, device):
        """
        Initializes the hidden state AND cell state with zeros.
        Returns a tuple: (h_0, c_0)
        """
        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (h_0, c_0)
