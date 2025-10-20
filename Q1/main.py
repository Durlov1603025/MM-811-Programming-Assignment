import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from datetime import datetime

from model import ConvNet
from train_eval import train_epoch, evaluate


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    if args.subset is not None and args.subset > 0:
        train_ds = Subset(train_ds, list(range(args.subset)))
        print(f"Using subset of {len(train_ds)} images for training")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = ConvNet(in_channels=1, num_classes=10).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = datetime.now()
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch:02d}: Train acc {train_acc:.2f}% | Test acc {test_acc:.2f}%")
        print(f"Train loss {train_loss:.4f} | Test loss {test_loss:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch time: {(datetime.now() - t0).total_seconds():.1f}s\n")

    print(f"Training complete! Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()

    main(args)
