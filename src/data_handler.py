import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class IAMLinesDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = self._load_samples(labels_file)

    def _load_samples(self, labels_file):
        samples = []

        with open(labels_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                # Skip lines with errors
                parts = line.strip().split()
                if len(parts) < 9 or parts[1] == "err":
                    continue

                line_id = parts[0]
                text = ' '.join(parts[8:])
                img_path = self._build_image_path(line_id)
                
                # If the image exists append the sample
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert("L")
                    samples.append((image, text))

        return samples

    def _build_image_path(self, line_id):
        pt1, pt2, _ = line_id.split('-')
        return os.path.join(self.img_dir, pt1, f"{pt1}-{pt2}", f"{line_id}.png")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, text = self.samples[idx]
        if self.transform:
            image = self.transform(image)
            
        return image, text

class LabelConverter:
    def __init__(self, vocab):
        # Initialize vocabulary with a blank token for CTC decoding
        self.vocab = ['<blank>'] + sorted(set(vocab))
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.blank = self.char_to_idx['<blank>']

    def encode(self, texts):
        # Convert list of strings to flat tensor of indices and their lengths
        targets = []
        lengths = []
        for text in texts:
            encoded = [self.char_to_idx[c] for c in text if c in self.char_to_idx]
            targets.extend(encoded)
            lengths.append(len(encoded))
        return torch.LongTensor(targets), torch.LongTensor(lengths)

    def decode(self, preds):
        # Greedy CTC decoding: collapse repeats and remove blanks
        results = []

        for seq in preds.argmax(2).cpu().numpy():
            prev = self.blank
            text = ''
            for idx in seq:
                if idx != self.blank and idx != prev:
                    text += self.idx_to_char.get(idx, '')
                prev = idx
            results.append(text)

        return results

def get_vocab(dataset):
    path = "./datasets/vocab.txt"
    
    if os.path.exists(path):
        return set(open(path).read())
    
    print("--Creating vocabulary...")
    vocab = {c for _, t in dataset for c in t}
    open(path, "w").write("".join(sorted(vocab)))
    return vocab

def get_iam_dataset(transform):
    print("Loading IAM dataset...")
    return IAMLinesDataset("datasets/lines", "datasets/lines.txt", transform)