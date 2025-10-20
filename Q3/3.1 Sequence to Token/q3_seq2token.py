import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

from autoencoder_model import AutoEncoder


# ============================================================================
#  1. Dataset: Builds autoregressive (prefix → next_token) pairs
# ============================================================================
class LatentSequenceDataset(Dataset):
    def __init__(self, autoencoder, images, device):
        self.device = device
        autoencoder.eval()
        
        # Extract binary latent codes
        with torch.no_grad():
            imgs_normalized = images.to(device) * 2 - 1  
            _, latents, _ = autoencoder(imgs_normalized)
        
        self.latents = latents.view(latents.size(0), -1)  # (N, 48), values {-1, +1}
        self.tokens = ((self.latents + 1) / 2).long()      
        
        # Create autoregressive training pairs
        seq_len = self.tokens.size(1)  # 48
        self.inputs, self.labels, self.lengths = [], [], []
        
        for seq in self.tokens:
            for i in range(1, seq_len):
                self.inputs.append(seq[:i])
                self.labels.append(seq[i])
                self.lengths.append(i)  
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        x = self.inputs[idx].float()
        y = self.labels[idx].float()
        length = self.lengths[idx]
        return x, y, length


def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    # Pad sequences to max length in this batch
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)
    
    return sequences_padded, labels, lengths

# ============================================================================
#  2. Model: LSTM-based Seq2Token Network
# ============================================================================
class Seq2TokenLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=2, embedding_dim=hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0  # Removed dropout to encourage memorization
        )
        
        # Output layers 
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, lengths=None):
        batch_size = x.size(0)
        
        # Embed tokens
        embedded = self.embedding(x.long())  
        if lengths is not None:
            packed = pack_padded_sequence(
                embedded, lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            # Use last hidden state
            output = hidden[-1]  
        else:
            # Process full sequence
            lstm_out, (hidden, cell) = self.lstm(embedded)
            output = hidden[-1] 
        
        # Predict next token
        pred = self.fc(output).squeeze(-1)  
        
        return pred

# ============================================================================
#  3. Training Loop
# ============================================================================
def train_seq2token(model, loader, device, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    
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
            
            # Forward pass
            pred = model(x, lengths)
            loss = loss_fn(pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predicted_class = (pred > 0.5).float()
            total_correct += (predicted_class == y).sum().item()
            total_samples += y.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{total_correct/total_samples:.3f}'
            })
        
        avg_loss = total_loss / len(loader)
        avg_acc = total_correct / total_samples
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
    
    return history

# ============================================================================
#  4. Sampling & Generation
# ============================================================================
def sample_sequences(model, device, num_samples=64, seq_len=48, temperature=1.0):
    model.eval()
    generated_tokens = torch.zeros(num_samples, seq_len, device=device)
    
    with torch.no_grad():
        for i in range(seq_len):
            if i == 0:
                # First token: start with empty sequence
                dummy = torch.zeros(num_samples, 1, device=device)
                probs = torch.ones(num_samples, device=device) * 0.5  
            else:
                # Use generated tokens so far
                current_seq = generated_tokens[:, :i]
                lengths = torch.full((num_samples,), i, device=device)
                probs = model(current_seq, lengths)
            
            # Apply temperature for diversity
            if temperature != 1.0:
                probs = torch.sigmoid(torch.logit(probs) / temperature)

            token = torch.bernoulli(probs)
            generated_tokens[:, i] = token
    
    return generated_tokens  # Returns {0, 1} tokens


def decode_latents(autoencoder, token_samples, device):
    autoencoder.eval()
    
    # Convert tokens
    latent_samples = token_samples * 2 - 1
    
    # Reshape to (N, 3, 4, 4) for decoder
    latent_samples = latent_samples.view(-1, 3, 4, 4)
    
    with torch.no_grad():
        imgs = autoencoder.decoder(latent_samples.to(device))
    
    return imgs


# ============================================================================
#  5. Evaluation: Check Memorization
# ============================================================================
def check_memorization(generated_tokens, train_tokens):
    diffs = (generated_tokens.unsqueeze(1) - train_tokens.unsqueeze(0)).abs().sum(dim=-1)
    
    # A perfect match has distance 0
    matches = (diffs == 0).any(dim=1).float()
    
    return matches.mean().item()


# ============================================================================
#  6. Visualization
# ============================================================================
def visualize_results(original_imgs, reconstructed_imgs, generated_imgs, 
                      train_imgs_decoded, save_path='q3_sec2token.png'):
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Original images
    grid = make_grid(original_imgs[:64], nrow=8, normalize=True, value_range=(-1, 1))
    axes[0].imshow(grid.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Original Images (First 64)', fontsize=14)
    axes[0].axis('off')
    
    # Reconstructed (through AE)
    grid = make_grid(reconstructed_imgs[:64], nrow=8, normalize=True, value_range=(-1, 1))
    axes[1].imshow(grid.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title('Reconstructed by AutoEncoder', fontsize=14)
    axes[1].axis('off')
    
    # Generated by Seq2Token
    grid = make_grid(generated_imgs[:64], nrow=8, normalize=True, value_range=(-1, 1))
    axes[2].imshow(grid.permute(1, 2, 0).cpu().numpy())
    axes[2].set_title('Generated by LSTM Seq2Token Model', fontsize=14)
    axes[2].axis('off')
    
    # Closest training images
    grid = make_grid(train_imgs_decoded[:64], nrow=8, normalize=True, value_range=(-1, 1))
    axes[3].imshow(grid.permute(1, 2, 0).cpu().numpy())
    axes[3].set_title('Closest Training Images (Memorization Check)', fontsize=14)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization -> {save_path}")
    plt.close()


def find_closest_images(generated_imgs, train_imgs):
    """Find closest training image for each generated image"""
    closest = []
    for gen_img in generated_imgs:
        dists = (gen_img - train_imgs).abs().reshape(train_imgs.size(0), -1).sum(dim=1)
        closest_idx = dists.argmin()
        closest.append(train_imgs[closest_idx])
    return torch.stack(closest)


# ============================================================================
#  7. Main Experiment
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--num_samples', type=int, default=128, help='Number of training images')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--num_generated', type=int, default=64, help='Number of images to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load AutoEncoder
    ae = AutoEncoder(use_binary=True).to(device)
    try:
        ae.load_state_dict(torch.load('autoencoder_best.pth', map_location=device))
    except FileNotFoundError:
        print("ERROR - autoencoder_best.pth not found!")
        return
    ae.eval()
    
    # Load MNIST
    tfm = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    mnist_full = datasets.MNIST(root='./data', train=True, download=True, transform=tfm)
    images = torch.stack([img for img, _ in list(mnist_full)[:args.num_samples]])
    
    # Create dataset
    dataset = LatentSequenceDataset(ae, images, device)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Train model
    model = Seq2TokenLSTM(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(device)
    
    history = train_seq2token(model, loader, device, epochs=args.epochs, lr=args.lr)
    
    torch.save(model.state_dict(), 'seq2token_lstm.pth')
    print(f"\nOK - Saved model -> seq2token_lstm.pth\n")
    
    # Generate sequences    
    generated_tokens = sample_sequences(
        model, device, 
        num_samples=args.num_generated,
        temperature=args.temperature
    )
    generated_imgs = decode_latents(ae, generated_tokens, device)
    
    # Reconstruct original
    with torch.no_grad():
        imgs_normalized = images[:args.num_generated].to(device) * 2 - 1
        reconstructed_imgs, _, _ = ae(imgs_normalized)
    
    # Check memorization
    train_tokens = dataset.tokens.float()
    memorization_ratio = check_memorization(
        generated_tokens.view(args.num_generated, -1),
        train_tokens
    )
    
    print(f"Memorization Ratio: {memorization_ratio * 100:.2f}%")
    print(f"({int(memorization_ratio * args.num_generated)}/{args.num_generated} "
          f"generated sequences match training data)\n")
    
    # Find closest images
    with torch.no_grad():
        all_train_imgs = images.to(device) * 2 - 1
        all_train_reconstructed, _, _ = ae(all_train_imgs)
    
    closest_imgs = find_closest_images(generated_imgs, all_train_reconstructed)
   
        
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY Sequence to Token")
    print("=" * 60)
    print(f"Training samples: {args.num_samples}")
    print(f"Training epochs: {args.epochs}")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history['acc'][-1]:.4f}")
    print(f"Generated samples: {args.num_generated}")
    print(f"Memorization rate: {memorization_ratio * 100:.2f}%")
    print("=" * 60)
    


if __name__ == '__main__':
    main()
