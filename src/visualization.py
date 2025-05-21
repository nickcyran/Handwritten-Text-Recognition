from torch import nn
import matplotlib.pyplot as plt
from collections import Counter
import math

# ---------- Generic Plotting Utilities ----------
def show_bar(x, y, title, xlabel, ylabel, figsize=(15, 4)):
    plt.figure(figsize=figsize)
    plt.bar(x, y, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def show_hist(data, bins, title, xlabel, ylabel, figsize=(10, 4)):
    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_lines(x, y_series, labels, title, ylabel, save_path=None):
    plt.figure(figsize=(6, 6)) 

    for y, label in zip(y_series, labels):
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()  

def show_image(image, title="Feature Map", figsize=(8, 2), cmap='viridis'):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ---------- Specific Visualizations ----------
def plot_text_lengths(dataset):
    lengths = [len(label) for _, label in dataset.samples]
    show_hist(lengths, bins=30, title="Histogram of Text Lengths", xlabel="Text Length", ylabel="Frequency")

def plot_char_distribution(dataset):
    all_chars = ''.join(c for _, label in dataset.samples for c in label if c != '|')
    counter = Counter(all_chars)
    chars, freqs = zip(*sorted(counter.items(), key=lambda x: -x[1]))
    show_bar(chars, freqs, title="Character Frequency Distribution", xlabel="Character", ylabel="Count")

def plot_metrics(metrics):
    epochs = list(range(1, len(metrics[0]) + 1))
    
    show_lines(
        x=epochs,
        y_series=[metrics[0], metrics[1]],
        labels=["Train Loss", "Val Loss"],
        title="Loss Over Epochs",
        ylabel="Loss",
        save_path="loss_over_epochs.png"
    )

    show_lines(
        x=epochs,
        y_series=[metrics[2], metrics[3]],
        labels=["CER", "WER"],
        title="Error Rates Over Epochs",
        ylabel="Error Rate",
        save_path="error_rates_over_epochs.png"
    )

def visualize_cnn_features(model, image_tensor):
    model.eval()
    x = image_tensor

    feature_maps = []
    titles = []

    for i, layer in enumerate(model.cnn):
        x = layer(x)
        if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
            feature_maps.append(x[0][0].cpu().numpy())
            titles.append(f"{i}: {layer.__class__.__name__}")

    # Plot all feature maps in a grid with 2 columns
    n = len(feature_maps)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 2), constrained_layout=True)

    # Flatten axes array for easy indexing
    axes = axes.flatten()

    for i, fmap in enumerate(feature_maps):
        axes[i].imshow(fmap, cmap='viridis', aspect='auto')
        axes[i].set_title(titles[i], fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.show()
