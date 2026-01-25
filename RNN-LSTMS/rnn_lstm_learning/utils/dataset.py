import torch
from torch.utils.data import Dataset

class CharacterDataset(Dataset):
    """
    A PyTorch Dataset that creates sliding window sequences for character-level modeling.
    
    Args:
        text (str): The entire training text.
        vocab (Vocabulary): The vocabulary object for encoding.
        seq_len (int): The length of the input sequence.
    """
    def __init__(self, text, vocab, seq_len=30):
        self.text = text
        self.vocab = vocab
        self.seq_len = seq_len
        
        # Convert the entire text to integers
        self.data = torch.tensor(self.vocab.encode(text), dtype=torch.long)
        breakpoint()

    def __len__(self):
        # We can produce len(text) - seq_len input-target pairs
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        """
        Returns:
            x (tensor): Input sequence of indices (seq_len,)
            y (tensor): Target sequence of indices (seq_len,) - shifted by 1
        """
        # Input sequence: from idx to idx + seq_len
        x = self.data[idx : idx + self.seq_len]
        
        # Target sequence: from idx + 1 to idx + seq_len + 1
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        
        return x, y
