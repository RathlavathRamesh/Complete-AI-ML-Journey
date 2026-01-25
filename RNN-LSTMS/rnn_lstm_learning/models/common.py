import torch

class Vocabulary:
    """
    Handles character-to-index and index-to-character mappings.
    """
    def __init__(self, text):
        # Find all unique characters in the text
        chars = sorted(list(set(text)))
        self.chars = chars
        self.vocab_size = len(chars)

        # Create mappings
        self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        """Converts a string of characters to a list of integers."""
        return [self.char_to_ix[ch] for ch in text]

    def decode(self, indices):
        """Converts a list (or tensor) of integers back to a string."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return ''.join([self.ix_to_char[ix] for ix in indices])

    def __len__(self):
        return self.vocab_size
