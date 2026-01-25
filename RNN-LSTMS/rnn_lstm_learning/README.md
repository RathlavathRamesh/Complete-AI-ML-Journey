# RNN vs LSTM: A Practical Breakdown

## Background
Built by a 2025 CSE Graduate and Associate AI Engineer at Centific. 
I have deep experience with Transformers (having built a 45M parameter GPT from scratch), but I built this project to return to first principles and practically understand the engineering mechanics of RNNs and LSTMs.

## Project Goal
This is not a toy project. It is a clean, modular, and educational implementation designed to demonstrate:
1.  **Recurrence**: How simple RNNs handle sequences.
2.  **Long-Term Dependencies**: Why RNNs fail at long sequences due to vanishing gradients.
3.  **The LSTM Solution**: How gating mechanisms (input, forget, output) allow LSTMs to retain context over longer distances.

## Key Learnings
- **Transformers vs RNNs**: While this project focuses on RNN/LSTM, it highlights why Transformers took over. RNNs process data sequentially (slow, hard to parallelize), whereas Transformers process sequences in parallel using attention.
- **Vanishing Gradients**: You will likely see the Simple RNN struggle to maintain coherent sentences structurally compared to the LSTM, or it might just repeat patterns if the sequence is too long for its small memory.

## How to Run

1.  **Install Dependencies**
    ```bash
    pip install torch
    ```

2.  **Train the RNN**
    ```bash
    python train_rnn.py
    ```
    *Trains for 200 epochs on a tiny dataset to ensure overfitting/memorization.*

3.  **Train the LSTM**
    ```bash
    python train_lstm.py
    ```

4.  **Compare Results**
    ```bash
    python compare_models.py
    ```
    *Generates text side-by-side from the same seed.*

## Project Structure
- `data/`: Contains the tiny training corpus.
- `models/`: Modular PyTorch implementations of `SimpleRNN` and `SimpleLSTM`.
- `utils/`: Data pipelines and training loops.
