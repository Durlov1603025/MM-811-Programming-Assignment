"""
Question 4: Text-based Autoregressive Memorization Model
=========================================================
Replaces image dataset with TEXT dataset and reproduces memorization experiments.

Demonstrates:
- Autoregressive model on text sequences (character-level)
- Memorization with extremely limited data (can work with just 2 sentences!)
- Same concepts as Q3 but applied to text domain

Usage:
    python q4_text_memorization.py --epochs 200 --num_samples 5
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


# ============================================================================
#  1. Text Dataset
# ============================================================================
# Sample text data 
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is the study of computer algorithms.",
    "Deep neural networks can learn complex patterns.",
    "Artificial intelligence transforms how we live.",
    "Python is a popular programming language.",
    "Data science combines statistics and computing.",
    "Natural language processing analyzes human language.",
    "Computer vision enables machines to see.",
    "Reinforcement learning trains agents through rewards.",
    "Generative models create new data samples.",
]


class TextSequenceDataset(Dataset):
    def __init__(self, texts, max_texts=None):
        if max_texts:
            texts = texts[:max_texts]
        
        self.texts = texts
        
        # Build vocabulary (character-level)
        all_chars = set(''.join(texts))
        self.char2idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        
        print(f"Dataset Info:")
        print(f"  - Number of texts: {len(texts)}")
        print(f"  - Vocabulary size: {self.vocab_size}")
        print(f"  - Characters: {sorted(all_chars)[:20]}...")
        
        # Convert texts to token sequences
        self.token_sequences = []
        for text in texts:
            tokens = [self.char2idx[c] for c in text]
            self.token_sequences.append(torch.tensor(tokens))
        
        # Create autoregressive training pairs
        self.inputs, self.labels, self.lengths = [], [], []
        for seq in self.token_sequences:
            for i in range(1, len(seq)):
                self.inputs.append(seq[:i])
                self.labels.append(seq[i])
                self.lengths.append(i)
        
        self.max_len = max(len(seq) for seq in self.token_sequences) - 1
        
        print(f"  - Training pairs: {len(self.inputs)}")
        print(f"  - Max sequence length: {self.max_len}\n")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        x = self.inputs[idx].float()
        y = self.labels[idx].long()
        length = self.lengths[idx]
        
        # Pad to max_len
        pad_len = self.max_len - len(x)
        x = F.pad(x, (0, pad_len), value=0)
        
        return x, y, length
    
    def decode_sequence(self, token_seq):
        """Convert token sequence back to text"""
        chars = [self.idx2char[int(t)] for t in token_seq if int(t) in self.idx2char]
        return ''.join(chars)


def collate_fn(batch):
    """Custom collate for variable-length sequences"""
    sequences, labels, lengths = zip(*batch)
    sequences_padded = torch.stack(sequences)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)
    return sequences_padded, labels, lengths


# ============================================================================
#  2. Model: Character-level LSTM for Text
# ============================================================================
class TextAutoRegressiveLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0  
        )
        
        # Output layers - produce distribution over vocab
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)  # Output: logits for each character
        )
    
    def forward(self, x, lengths=None):
        # Embed
        embedded = self.embedding(x.long())
        
        # LSTM
        if lengths is not None:
            from torch.nn.utils.rnn import pack_padded_sequence
            packed = pack_padded_sequence(
                embedded, lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            output = hidden[-1]
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
            output = hidden[-1]
        
        # Predict distribution over vocab
        logits = self.fc(output)  
        probs = F.softmax(logits, dim=1) 
        
        return logits, probs


# ============================================================================
#  3. Training
# ============================================================================
def train_text_model(model, loader, device, epochs=200, lr=1e-3):
    """Train with CrossEntropyLoss (for multi-class distribution)"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    
    history = {'loss': [], 'acc': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for x, y, lengths in pbar:
            x, y = x.to(device), y.to(device)
            lengths = lengths.to(device)
            
            # Forward
            logits, probs = model(x, lengths)
            loss = loss_fn(logits, y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predicted = logits.argmax(dim=1)
            total_correct += (predicted == y).sum().item()
            total_samples += y.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{total_correct/total_samples:.3f}'
            })
        
        scheduler.step()
        
        avg_loss = total_loss / len(loader)
        avg_acc = total_correct / total_samples
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        
        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    
    return history


# ============================================================================
#  4. Generation (Autoregressive Text Generation)
# ============================================================================
def generate_text_sequences(model, dataset, device, num_samples=10, 
                            max_len=50, temperature=0.5):
    model.eval()
    
    generated_texts = []
    
    with torch.no_grad():
        for sample_idx in range(num_samples):
            generated_tokens = []
            
            # Use greedy decoding (argmax) for deterministic generation
            start_char_idx = sample_idx % dataset.vocab_size
            generated_tokens.append(start_char_idx)
            
            for i in range(1, max_len):
                current_seq_input = torch.tensor([generated_tokens], device=device).float()
                length = torch.tensor([len(generated_tokens)], device=device)
                
                logits, probs = model(current_seq_input, length)
                
                # Use temperature sampling (low temp = more deterministic)
                if temperature < 0.1:
                    # Greedy: always pick most likely
                    token = logits.argmax(dim=1).item()
                else:
                    # Sample from distribution with temperature
                    logits_scaled = logits / temperature
                    probs_scaled = F.softmax(logits_scaled, dim=1)
                    token = torch.multinomial(probs_scaled[0], num_samples=1).item()
                
                generated_tokens.append(token)
                
                # Stop at period
                if dataset.idx2char.get(token, '') == '.':
                    break
            
            # Decode to text
            text = dataset.decode_sequence(generated_tokens)
            generated_texts.append(text)
    
    return generated_texts, None


# ============================================================================
#  5. Evaluation: Text Memorization
# ============================================================================
def check_text_memorization(generated_texts, train_texts):
    """
    Check how many generated texts exactly match training texts.
    """
    matches = 0
    for gen_text in generated_texts:
        if gen_text in train_texts:
            matches += 1
    
    return matches / len(generated_texts)


def visualize_text_results(train_texts, generated_texts, memorization_ratio,
                           save_path='q4_text_results.txt'):
    """
    Save text results to file for visualization.
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("QUESTION 4: TEXT MEMORIZATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Training Texts ({len(train_texts)} total):\n")
        f.write("-"*70 + "\n")
        for i, text in enumerate(train_texts, 1):
            f.write(f"{i}. {text}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write(f"Generated Texts ({len(generated_texts)} total):\n")
        f.write("-"*70 + "\n")
        for i, text in enumerate(generated_texts, 1):
            match = "✓ MATCH" if text in train_texts else "✗ Novel"
            f.write(f"{i}. {text} [{match}]\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write(f"Memorization Rate: {memorization_ratio*100:.2f}%\n")
        f.write("="*70 + "\n")
    
    # print(f"Saved text results -> {save_path}")


# ============================================================================
#  6. Main Experiment
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of training texts (can be as low as 2)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=256, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=3, help='LSTM layers')
    parser.add_argument('--num_generated', type=int, default=20, help='Texts to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--custom_text', type=str, nargs='+', help='Custom text samples')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # -------------------------------------------------------------------------
    # Step 1: Prepare Text Data
    # -------------------------------------------------------------------------
    # print("=" * 70)
    # print("STEP 1: Preparing Text Dataset")
    # print("=" * 70)
    
    # Use custom texts if provided, otherwise use samples
    if args.custom_text:
        train_texts = args.custom_text
    else:
        train_texts = SAMPLE_TEXTS[:args.num_samples]
    
    # print(f"Training on {len(train_texts)} text samples:")
    # print("-" * 70)
    # for i, text in enumerate(train_texts, 1):
    #     print(f"{i}. {text}")
    # print("-" * 70 + "\n")
    
    # -------------------------------------------------------------------------
    # Step 2: Create Dataset
    # -------------------------------------------------------------------------
    # print("=" * 70)
    # print("STEP 2: Creating autoregressive dataset")
    # print("=" * 70)
    
    dataset = TextSequenceDataset(train_texts)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Train Model
    # -------------------------------------------------------------------------
    # print("=" * 70)
    # print("STEP 3: Training Autoregressive Text Model")
    # print("=" * 70)
    
    model = TextAutoRegressiveLSTM(
        vocab_size=dataset.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # print(f"Output: Histogram over {dataset.vocab_size} characters\n")
    
    history = train_text_model(model, loader, device, epochs=args.epochs, lr=args.lr)
    
    torch.save({
        'model_state': model.state_dict(),
        'char2idx': dataset.char2idx,
        'idx2char': dataset.idx2char,
        'vocab_size': dataset.vocab_size
    }, 'text_memorization.pth')
    print(f"\nSaved model -> text_memorization.pth\n")
    
    # -------------------------------------------------------------------------
    # Step 4: Generate Text
    # -------------------------------------------------------------------------
    # print("=" * 70)
    # print("STEP 4: Generating text sequences")
    # print("=" * 70)
    
    generated_texts, generated_seqs = generate_text_sequences(
        model, dataset, device,
        num_samples=args.num_generated,
        max_len=max(len(t) for t in train_texts) + 10,
        temperature=args.temperature
    )
    
    # print(f"Generated {len(generated_texts)} text samples:\n")
    # for i, text in enumerate(generated_texts[:10], 1):
    #     match = "✓" if text in train_texts else "✗"
    #     print(f"{i}. [{match}] {text}")
    
    # if len(generated_texts) > 10:
    #     print(f"... ({len(generated_texts) - 10} more)")
    # print()
    
    # -------------------------------------------------------------------------
    # Step 5: Check Memorization
    # -------------------------------------------------------------------------
    # print("=" * 70)
    # print("STEP 5: Evaluating memorization")
    # print("=" * 70)
    
    memo_ratio = check_text_memorization(generated_texts, train_texts)
    
    matches_count = sum(1 for t in generated_texts if t in train_texts)
    print(f"Memorization Ratio: {memo_ratio * 100:.2f}%")
    print(f"({matches_count}/{len(generated_texts)} generated texts match training)\n")
    
    # -------------------------------------------------------------------------
    # Step 6: Visualize Results
    # -------------------------------------------------------------------------
    # print("=" * 70)
    # print("STEP 6: Saving results")
    # print("=" * 70)
    
    visualize_text_results(
        train_texts,
        generated_texts,
        memo_ratio,
        save_path='q4_text_results.txt'
    )
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY (TEXT MEMORIZATION)")
    print("=" * 50)
    print(f"Training texts: {len(train_texts)}")
    print(f"Vocabulary size: {dataset.vocab_size} characters")
    print(f"Training epochs: {args.epochs}")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history['acc'][-1]:.4f}")
    print(f"Generated samples: {len(generated_texts)}")
    print(f"Memorization rate: {memo_ratio * 100:.2f}%")
    print("=" * 50)


if __name__ == '__main__':
    main()

