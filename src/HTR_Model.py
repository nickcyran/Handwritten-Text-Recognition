import os
import time
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm
import editdistance
import matplotlib.pyplot as plt
import shutil
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from src.data_handler import LabelConverter, get_iam_dataset, get_vocab
from src.preprocessing import dataset_transform, evaluation_transform
from src.visualization import plot_metrics, visualize_cnn_features

# Enable cuDNN auto-tuner for optimized performance
torch.backends.cudnn.benchmark = True

# Set high precision for float32 matrix multiplications for
torch.set_float32_matmul_precision('high')

# Select device: CUDA if available else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CRNN(nn.Module):
    def __init__(self, num_classes, H):
        super().__init__()
        
        def conv_block(in_channels, out_channels, pool=None):
            return [
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                *([nn.MaxPool2d(pool, pool)] if pool else [])
            ]

        # ----- CNN Feature Extractor -----
        self.cnn = nn.Sequential(
            *conv_block(1, 64),
            *conv_block(64, 64, 2),
            *conv_block(64, 128, 2),
            *conv_block(128, 256),
            *conv_block(256, 256, (2,1)),
            nn.Dropout(0.3)
        )
        
        # ----- Linear projection -----
        self.linear_before_lstm = nn.Linear(256 * (H // 8), 256)
        
        # ----- Bidirectional LSTM -----
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=256, num_layers=3, 
            bidirectional=True, batch_first=True, dropout=0.3
        )
        
        # ----- Final Classification Layer -----
        self.fc = nn.Linear(512, num_classes) 

    def forward(self, x):
        # Input: (B, 1, H, W)
        x = self.cnn(x)                                 # -> (B, C, H', W')
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2).reshape(B, W, -1)     # -> (B, W, C*H)
        x = self.linear_before_lstm(x)                  # -> (B, W, 256)
        x, _ = self.lstm(x)                             # -> (B, W, 512)
        x = self.fc(x)                                  # -> (B, W, num_classes)
        return x.permute(1, 0, 2)                       # -> (T, B, C) for CTC loss

class ModelHTR:
    def __init__(self):
        # Define transforms
        self.transform = dataset_transform()
        self.eval_transform = evaluation_transform()

        # Load and split dataset
        dataset = get_iam_dataset(self.transform)
        val_size = int(0.1 * len(dataset))
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        
        # Setup converter and model
        self.converter = LabelConverter(get_vocab(dataset))
        self.model = CRNN(num_classes=len(self.converter.vocab), H=64).to(device)

        # Compile model if possible
        if hasattr(torch, "compile") and torch.cuda.is_available() and shutil.which("cl.exe"):
            self.model = torch.compile(self.model)

    def train(self, epochs=75, patience=8, save_path="best_model.pth"):
        print("Training...")
        start_time = time.time()
        
        # Data loaders
        train_ldr = create_dataloader(self.train_dataset, shuffle=True)
        val_ldr = create_dataloader(self.val_dataset, shuffle=False)

        # Train optimization setup
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')

        # Information vars
        best_loss = float('inf')
        epochs_without_improvement = 0
        metrics = [[],[],[],[]]    # [train_losses, val_losses, cer, wer]
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_ldr, desc=f"Epoch {epoch + 1}")

            for images, txts in pbar:
                images, targs, targ_lengths = move_to_device(images, txts, self.converter)
                
                # Use amp if supported 
                with autocast('cuda' if device.type == 'cuda' else 'cpu') :
                    preds = self.model(images)
                    input_lengths = torch.full([preds.size(1)], preds.size(0), dtype=torch.long, device=device)
                    loss = criterion(preds.log_softmax(2), targs, input_lengths, targ_lengths)

                # Backpropagation and optimizer
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            # Validation
            avg_train_loss = epoch_loss / len(train_ldr)
            val_loss, cer, wer = self.evaluate(val_ldr, criterion)
            print(f"Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, CER={cer:.4f}, WER={wer:.4f}")
            
            # Learning rate scheduling & metrics
            scheduler.step(val_loss)
            metrics[0].append(avg_train_loss)
            metrics[1].append(val_loss)
            metrics[2].append(cer)
            metrics[3].append(wer)

            # Early stopping mechanism
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model to: {save_path}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Early stopping triggered.")
                    break
        
        # Display time took to train
        hrs, rem = divmod(int((time.time() - start_time)), 3600)
        mins, secs = divmod(rem, 60)
        print(f"\nTotal training time: {hrs}h {mins}m {secs}s")

        plot_metrics(metrics)

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        loss_tot = 0
        char_edits = 0
        char_total = 0
        word_edits = 0
        word_total = 0

        with torch.no_grad():
            for images, txts in tqdm(dataloader, desc="Evaluating"):
                images, targs, targ_lengths = move_to_device(images, txts, self.converter)

                # Forward pass
                preds = self.model(images)
                input_lengths = torch.full([preds.size(1)], preds.size(0), dtype=torch.long, device=device)

                # Compute loss
                loss = criterion(preds.log_softmax(2), targs, input_lengths, targ_lengths)
                loss_tot += loss.item()

                # Decode predictions
                decoded = self.converter.decode(preds.permute(1, 0, 2))

                # Compute CER and WER
                for pred, truth in zip(decoded, txts):
                    pred_clean = pred.replace('|', ' ').strip()
                    truth_clean = truth.replace('|', ' ').strip()

                    # CER
                    char_edits += editdistance.eval(pred_clean, truth_clean)
                    char_total += len(truth_clean)

                    # WER
                    pred_words = pred_clean.split()
                    truth_words = truth_clean.split()
                    word_edits += editdistance.eval(pred_words, truth_words)
                    word_total += len(truth_words)

        self.model.train()
        avg_loss = loss_tot / len(dataloader)
        cer = (char_edits / char_total) if char_total > 0 else 0
        wer = (word_edits / word_total) if word_total > 0 else 0
        return avg_loss, cer, wer

    def predict(self, image_path, visualize=False):
        self.model.eval()

        with torch.no_grad():
            # Load original image (for display)
            pil_img = Image.open(image_path).convert("L")

            # Transform image for model input
            tensor = self.eval_transform(pil_img).unsqueeze(0).to(device)

            preds = self.model(tensor)  # (T, B, C)
            decoded = self.converter.decode(preds.permute(1, 0, 2))
            text = decoded[0].replace('|', ' ').strip()
            
            if visualize:
                visualize_cnn_features(self.model, tensor)

                # Show image with predicted text
                plt.figure(figsize=(10, 3))
                plt.imshow(tensor[0][0].cpu().numpy(), cmap='gray')
                plt.axis('off')
                plt.title(f"Prediction: {text}", fontsize=12)
                plt.tight_layout()
                plt.show()

            return text

    def load(self, path):
        state_dict = torch.load(path, map_location=device)

        # If compiled model was saved, unwrap the keys
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)

# ---------------------------------------------------------------- #
#                         Helper Functions
# ---------------------------------------------------------------- #
def create_dataloader(
    dataset, shuffle, 
    batch_size=32, num_workers=os.cpu_count(), 
    pin_memory=True, persistent_workers=True
) -> DataLoader:   
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

# Moves input data into the device
def move_to_device(images, txts, converter):
    images = images.to(device, non_blocking=True)
    targs, targ_lengths = converter.encode(txts)
    targs = targs.to(device, non_blocking=True) 
    targ_lengths = targ_lengths.to(device, non_blocking=True)
    return images, targs, targ_lengths