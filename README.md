# Line-Level Handwritten Text Recognition (HTR)

This project implements a deep learning-based offline handwritten text recognition (HTR) system using a CRNN (Convolutional Recurrent Neural Network) with CTC (Connectionist Temporal Classification) loss. It is designed for line-level transcription on the IAM dataset.


## Project Structure

```
./project/
    ├── HTR.py              # CLI entry point
    ├── datasets/
    │   ├── lines/               # IAM line images
    │   └── lines.txt            # IAM line text annotations
    └── src/
        ├── HTR_Model.py         # CRNN-CTC architecture and logic
        ├── data_handler.py      # Dataset loader and label encoding/decoding
        ├── preprocessing.py     # Preprocessing and transformation utilities
        └── visualization.py     # Tools for visualizing data

```

## Setup
1. Unzip `project.zip` and navigate to the directory `.../project/`.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

You need the following packages:
- `torch`
- `torchvision`
- `tqdm`
- `Pillow`
- `editdistance`
- `matplotlib`

‼️**IMPORTANT**‼️\
3. Ensure that project structure matches diagram. 


---

## Usage

Run the CLI using:

```bash
python HTR.py [OPTIONS]
```
###  CLI Options

| Argument         | Description |
|------------------|-------------|
| `--mode train`   | Train the model from scratch or from checkpoint. |
| `--mode predict` | Run prediction on an image or folder of images. |
| `--model PATH`   | Path to a `.pth` model checkpoint to load. Defaults to `best_model.pth`. |
| `--save_path PATH` | Path to save the trained model. Defaults to same as `--model`. |
| `--image_path PATH` | Image path (or directory) to use in prediction mode. |
| `--epochs N`     | Number of training epochs (default: 75). |
| `--visualize`    | Enable visualizations (feature maps, data distributions). |
| `--eval`         | Evaluate the model on validation data. |

### Examples

#### Train a model:
```
python HTR.py --mode train --epochs 60 --save_path htr_model.pth
```

#### Predict a single image:
```
python HTR.py --mode predict --image_path sample.png --model htr_model.pth
```

#### Predict all images in a folder:
```
python HTR.py --mode predict --image_path ./test_images/ --model htr_model.pth
```

#### Evaluate a model:
```
python HTR.py --eval --model htr_model.pth
```

#### Visualize dataset statistics:
```
python HTR.py --mode train --visualize
```

---

## Metrics

The system evaluates on:
- **CER (Character Error Rate)**
- **WER (Word Error Rate)**

Metric plots are saved as:
- `loss_over_epochs.png`
- `error_rates_over_epochs.png`

