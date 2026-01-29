import os
import cv2
import torch
import numpy as np
import pandas as pd
import string
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Levenshtein import distance as levenshtein
import random
import matplotlib.pyplot as plt

# ============================================
# CONFIGURATION
# ============================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

IMG_HEIGHT = 32
BATCH_SIZE = 32
EPOCHS = 60
INITIAL_LR = 1e-3

# Character set
CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase
char2idx = {c: i + 1 for i, c in enumerate(CHARS)}
idx2char = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(char2idx) + 1


# ============================================
# DATA PREPROCESSING
# ============================================

def preprocess_image(img_path, img_height=32, augment=False):
    """
    Robust preprocessing that won't crash.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read: {img_path}")
        
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # AUGMENTATION - simplified to avoid crashes
        if augment:
            # Ensure uint8 type
            img = img.astype(np.uint8)
            
            # Brightness/contrast (safe)
            if random.random() < 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.randint(-15, 15)
                img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
            # Translation (safe)
            if random.random() < 0.4:
                h, w = img.shape
                dx = int(random.uniform(-0.05, 0.05) * w)
                dy = int(random.uniform(-0.05, 0.05) * h)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, M, (w, h), 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=255)
            
            # Blur (safe - ensure uint8 first)
            if random.random() < 0.25:
                img = img.astype(np.uint8)
                img = cv2.GaussianBlur(img, (3, 3), 0)
            
            # Noise (safe)
            if random.random() < 0.2:
                noise = np.random.randn(*img.shape) * 5
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Resize
        h, w = img.shape
        new_w = max(1, int(w * (img_height / h)))
        img = cv2.resize(img, (new_w, img_height), interpolation=cv2.INTER_LINEAR)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        
        return img
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        # Return a blank image rather than crashing
        blank = np.zeros((img_height, 100), dtype=np.float32)
        return blank


def load_annotations(csv_path, img_root):
    """Load with validation."""
    df = pd.read_csv(csv_path)
    samples = []
    
    for _, row in df.iterrows():
        img_name = os.path.basename(row["ImgName"].strip())
        img_path = os.path.join(img_root, img_name)
        
        if os.path.exists(img_path):
            label = str(row["GroundTruth"])
            # Filter out invalid characters
            if all(c in char2idx for c in label):
                samples.append((img_path, label))
    
    return samples


def encode_label(text):
    return [char2idx[c] for c in text if c in char2idx]


class IIIT5KDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = preprocess_image(img_path, augment=self.augment)
            label_encoded = encode_label(label)
            
            if len(label_encoded) == 0:
                label_encoded = [1]  # Dummy label if empty
            
            image = torch.tensor(image).unsqueeze(0)
            label = torch.tensor(label_encoded)
            
            return image, label
        
        except Exception as e:
            print(f"Error in __getitem__ for {img_path}: {e}")
            # Return dummy data
            dummy_img = torch.zeros(1, 32, 100)
            dummy_label = torch.tensor([1])
            return dummy_img, dummy_label


def collate_fn(batch):
    """Safe collate function."""
    images, labels = zip(*batch)
    
    # Pad images
    heights = images[0].shape[1]
    widths = [img.shape[2] for img in images]
    max_width = max(widths)
    
    padded_images = torch.zeros(len(images), 1, heights, max_width)
    for i, img in enumerate(images):
        w = img.shape[2]
        padded_images[i, :, :, :w] = img
    
    # Concatenate labels
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = torch.cat(labels)
    
    return padded_images, labels, label_lengths, torch.tensor(widths)


# ============================================
# MODEL
# ============================================

class ImprovedCRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            
            nn.Conv2d(512, hidden_size, 2, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )
        
        self.rnn = nn.LSTM(
            hidden_size, hidden_size, 2,
            bidirectional=True, batch_first=True, dropout=0.3
        )
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        features = self.cnn(x)
        features = features.squeeze(2).permute(0, 2, 1)
        seq, _ = self.rnn(features)
        seq = self.dropout(seq)
        return self.fc(seq)


def greedy_decode(logits):
    """Fixed decoder."""
    preds = logits.argmax(2)
    results = []
    
    for seq in preds:
        decoded = []
        prev = None
        for p in seq:
            p = p.item()
            if p == 0:
                prev = None
                continue
            if p != prev:
                decoded.append(idx2char[p])
                prev = p
        results.append("".join(decoded))
    
    return results


# ============================================
# TRAINING
# ============================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for imgs, labels, label_lens, _ in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        logits = model(imgs)
        log_probs = logits.log_softmax(2)
        
        input_lens = torch.full(
            (logits.size(0),), logits.size(1), dtype=torch.long
        )
        
        loss = criterion(
            log_probs.permute(1, 0, 2),
            labels, input_lens, label_lens
        )
        
        if not torch.isfinite(loss):
            print("Warning: Invalid loss, skipping batch")
            continue
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    edit_dist = []
    
    with torch.no_grad():
        for imgs, labels, label_lens, _ in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = greedy_decode(logits)
            
            gt_texts = []
            idx = 0
            for l in label_lens:
                gt = "".join(idx2char[i.item()] for i in labels[idx:idx+l])
                gt_texts.append(gt)
                idx += l
            
            for gt, pred in zip(gt_texts, preds):
                if gt == pred:
                    correct += 1
                edit_dist.append(levenshtein(gt, pred) / max(len(gt), 1))
                total += 1
    
    return {
        "word_accuracy": correct / total,
        "mean_edit_distance": sum(edit_dist) / len(edit_dist)
    }


# ============================================
# MAIN
# ============================================

def main():
    # Load data
    print("\nLoading data...")
    train_samples = load_annotations("IIIT5K/traindata.csv", "IIIT5K/train")
    test_samples = load_annotations("IIIT5K/testdata.csv", "IIIT5K/test")
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")
    
    # Create datasets
    train_ds = IIIT5KDataset(train_samples, augment=True)
    test_ds = IIIT5KDataset(test_samples, augment=False)
    
    # CRITICAL: num_workers=0
    train_loader = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    # Model
    print("\nBuilding model...")
    model = ImprovedCRNN(NUM_CLASSES).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), INITIAL_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', factor=0.5, patience=5
    )
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Training
    best_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_accs = []
    
    print(f"\nTraining for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        train_losses.append(loss)
        
        metrics = evaluate(model, test_loader, DEVICE)
        val_accs.append(metrics['word_accuracy'])
        scheduler.step(metrics['word_accuracy'])
        
        print(f"Loss: {loss:.4f} | Acc: {metrics['word_accuracy']*100:.2f}% | "
              f"Edit: {metrics['mean_edit_distance']:.4f}")
        
        if metrics['word_accuracy'] > best_acc:
            best_acc = metrics['word_accuracy']
            patience_counter = 0
            torch.save(model.state_dict(), "crnn_robust_best.pth")
            print(f"✅ Best model saved! ({best_acc*100:.2f}%)")
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            print("Early stopping")
            break
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(train_losses)
    ax[0].set_title('Loss')
    ax[1].plot([a*100 for a in val_accs])
    ax[1].set_title('Accuracy %')
    plt.savefig('training_robust.png')
    
    print(f"\nDone! Best: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()