import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

from autoencoder_model import AutoEncoder


# ============================================================================
#  1. Dataset
# ============================================================================
class SequenceDataset(Dataset):
    """Full binary sequences"""
    def __init__(self, autoencoder, images, device):
        self.device = device
        autoencoder.eval()
        
        with torch.no_grad():
            imgs_normalized = images.to(device) * 2 - 1
            _, latents, _ = autoencoder(imgs_normalized)
        
        self.latents = latents.view(latents.size(0), -1)
        self.sequences = ((self.latents + 1) / 2).float()  # {0, 1}
        
        print(f"Dataset: {len(self.sequences)} sequences of length {self.sequences.size(1)}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


# ============================================================================
#  2. Residual FC Block
# ============================================================================
class ResidualFC(nn.Module):
    """Residual fully-connected block"""
    def __init__(self, dim, hidden_scale=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim * hidden_scale),
            nn.LayerNorm(dim * hidden_scale),
            nn.GELU(),
            nn.Linear(dim * hidden_scale, dim),
        )
    
    def forward(self, x):
        return x + self.block(x)


# ============================================================================
#  3. Seq2Seq Model
# ============================================================================
class Seq2SeqFC(nn.Module):
    def __init__(self, seq_len=48, hidden_dim=256, num_blocks=12):
        super().__init__()
        
        self.seq_len = seq_len
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualFC(hidden_dim, hidden_scale=4)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, seq_len),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Project to hidden dim
        h = self.input_proj(x)
        
        # Process through residual blocks
        for block in self.blocks:
            h = block(h)
        
        # Project to output
        out = self.output_proj(h)
        
        return out


# ============================================================================
#  4. Training with Noise/Masking
# ============================================================================
def train_seq2seq_fc(model, loader, device, epochs=200, lr=1e-3, noise_prob=0.1):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.BCELoss()
    
    history = {'loss': [], 'acc': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for sequences in pbar:
            sequences = sequences.to(device)
            
            # Add noise to input (denoising autoencoder)
            if noise_prob > 0:
                noise_mask = torch.rand_like(sequences) < noise_prob
                noisy_sequences = sequences.clone()
                noisy_sequences[noise_mask] = 1 - noisy_sequences[noise_mask]
            else:
                noisy_sequences = sequences
            
            # Predict original sequence
            pred = model(noisy_sequences)
            loss = loss_fn(pred, sequences)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predicted = (pred > 0.5).float()
            total_correct += (predicted == sequences).sum().item()
            total_tokens += sequences.numel()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{total_correct/total_tokens:.3f}'
            })
        
        scheduler.step()
        
        avg_loss = total_loss / len(loader)
        avg_acc = total_correct / total_tokens
        history['loss'].append(avg_loss)
        history['acc'].append(avg_acc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return history


# ============================================================================
#  5. Generation 
# ============================================================================
def generate_sequences_direct(model, device, num_samples=64, seq_len=48):
    model.eval()
    
    with torch.no_grad():
        # Start with random sequences
        sequences = torch.rand(num_samples, seq_len, device=device)
        sequences = (sequences > 0.5).float()
        
        # Let model refine (multiple passes for convergence)
        for _ in range(10):
            sequences = model(sequences)
            sequences = (sequences > 0.5).float()
    
    return sequences


def decode_sequences(autoencoder, token_sequences, device):
    autoencoder.eval()
    latent_sequences = token_sequences * 2 - 1
    latent_sequences = latent_sequences.view(-1, 3, 4, 4)
    
    with torch.no_grad():
        imgs = autoencoder.decoder(latent_sequences.to(device))
    
    return imgs


# ============================================================================
#  6. Evaluation & Visualization
# ============================================================================
def check_memorization(generated_seqs, train_seqs):
    """Check exact matches"""
    diffs = (generated_seqs.unsqueeze(1) - train_seqs.unsqueeze(0)).abs().sum(dim=-1)
    matches = (diffs == 0).any(dim=1).float()
    return matches.mean().item()


def visualize_results(original_imgs, reconstructed_imgs, generated_imgs, 
                      closest_imgs, save_path='q3_seq2seq_fc_results.png'):
    """Create 4-row comparison"""
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    grid = make_grid(original_imgs[:64], nrow=8, normalize=True, value_range=(-1, 1))
    axes[0].imshow(grid.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Original Images (First 64)', fontsize=14)
    axes[0].axis('off')
    
    grid = make_grid(reconstructed_imgs[:64], nrow=8, normalize=True, value_range=(-1, 1))
    axes[1].imshow(grid.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title('Reconstructed by AutoEncoder', fontsize=14)
    axes[1].axis('off')
    
    grid = make_grid(generated_imgs[:64], nrow=8, normalize=True, value_range=(-1, 1))
    axes[2].imshow(grid.permute(1, 2, 0).cpu().numpy())
    axes[2].set_title('Generated by Seq2Seq FC Model', fontsize=14)
    axes[2].axis('off')
    
    grid = make_grid(closest_imgs[:64], nrow=8, normalize=True, value_range=(-1, 1))
    axes[3].imshow(grid.permute(1, 2, 0).cpu().numpy())
    axes[3].set_title('Closest Training Images (Memorization Check)', fontsize=14)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization -> {save_path}")
    plt.close()


def find_closest_images(generated_imgs, train_imgs):
    closest = []
    for gen_img in generated_imgs:
        dists = (gen_img - train_imgs).abs().reshape(train_imgs.size(0), -1).sum(dim=1)
        closest_idx = dists.argmin()
        closest.append(train_imgs[closest_idx])
    return torch.stack(closest)


# ============================================================================
#  7. Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of training images')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_blocks', type=int, default=16, help='Number of residual blocks')
    parser.add_argument('--noise_prob', type=float, default=0.05, help='Noise probability for denoising')
    parser.add_argument('--num_generated', type=int, default=64, help='Number to generate')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load AutoEncoder
    # print("=" * 70)
    # print("STEP 1: Loading AutoEncoder")
    # print("=" * 70)
    
    ae = AutoEncoder(use_binary=True).to(device)
    try:
        ae.load_state_dict(torch.load('autoencoder_best.pth', map_location=device))
        print("OK - Loaded autoencoder_best.pth\n")
    except:
        print("ERROR - Can't load autoencoder")
        return
    ae.eval()
    
    # Load Data
    # print("=" * 70)
    # print("STEP 2: Preparing dataset")
    # print("=" * 70)
    
    tfm = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=tfm)
    images = torch.stack([img for img, _ in list(mnist)[:args.num_samples]])
    # print(f"Using {args.num_samples} MNIST images\n")
    
    # # Create Dataset
    # print("=" * 70)
    # print("STEP 3: Creating dataset")
    # print("=" * 70)
    
    dataset = SequenceDataset(ae, images, device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # print(f"Created {len(loader)} batches\n")
    
    # Train Model
    # print("=" * 70)
    # print("STEP 4: Training Seq2Seq FC Model")
    # print("=" * 70)
    
    model = Seq2SeqFC(
        seq_len=48,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks
    ).to(device)
    
    # print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    history = train_seq2seq_fc(
        model, loader, device,
        epochs=args.epochs,
        lr=args.lr,
        noise_prob=args.noise_prob
    )
    
    torch.save(model.state_dict(), 'seq2seq_fc.pth')
    # print(f"\nOK - Saved model -> seq2seq_fc.pth\n")
    
    # Generate
    # print("=" * 70)
    # print("STEP 5: Generating sequences")
    # print("=" * 70)
    
    generated_seqs = generate_sequences_direct(model, device, num_samples=args.num_generated)
    generated_imgs = decode_sequences(ae, generated_seqs, device)
    # print(f"Generated {args.num_generated} sequences\n")
    
    # # Reconstruct
    # print("=" * 70)
    # print("STEP 6: Reconstructing originals")
    # print("=" * 70)
    
    with torch.no_grad():
        imgs_normalized = images[:args.num_generated].to(device) * 2 - 1
        reconstructed_imgs, _, _ = ae(imgs_normalized)
    # print(f"Reconstructed {args.num_generated} images\n")
    
    # Memorization
    # print("=" * 70)
    # print("STEP 7: Evaluating memorization")
    # print("=" * 70)
    
    train_seqs = dataset.sequences
    memo_ratio = check_memorization(generated_seqs, train_seqs)
    
    print(f"Memorization Ratio: {memo_ratio * 100:.2f}%")
    print(f"({int(memo_ratio * args.num_generated)}/{args.num_generated} sequences match training)\n")
    
    # Closest images
    # print("=" * 70)
    # print("STEP 8: Finding closest training images")
    # print("=" * 70)
    
    with torch.no_grad():
        all_train_imgs = images.to(device) * 2 - 1
        all_train_reconstructed, _, _ = ae(all_train_imgs)
    
    closest_imgs = find_closest_images(generated_imgs, all_train_reconstructed)
    # print("Found closest matches\n")
    
    # Visualize
    # print("=" * 70)
    # print("STEP 9: Saving visualizations")
    # print("=" * 70)
    
    visualize_results(
        imgs_normalized,
        reconstructed_imgs,
        generated_imgs,
        closest_imgs,
        save_path='q3_seq2seq_fc_results.png'
    )
    
    save_image((generated_imgs + 1) / 2, 'q3_seq2seq_fc_generated.png', nrow=8)
    #print("OK - Saved images\n")
    
    # Summary
    print("=" * 50)
    print("SUMMARY (Sequence to Sequence)")
    print("=" * 50)
    print(f"Training samples: {args.num_samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history['acc'][-1]:.4f}")
    print(f"Memorization: {memo_ratio * 100:.2f}%")
    
    # if memo_ratio >= 0.80:
    #     print("\n✓✓✓ SUCCESS: >=80% memorization (FULL MARKS!)")
    # else:
    #     print(f"\n⚠ Below 80%. Suggestions:")
    #     print(f"  - Try --num_samples 32 (fewer samples)")
    #     print(f"  - Try --epochs 300 (more training)")
    #     print(f"  - Try --num_blocks 20 (larger model)")
    
    print("=" * 50)

if __name__ == '__main__':
    main()

