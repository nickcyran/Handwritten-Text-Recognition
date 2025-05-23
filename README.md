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

## Datasets  

This project uses the **IAM Handwriting Database** for training and testing the model. You'll need to download it to get started.

### Setting up IAMLines

1.  **Register and Download:**
    * First, you need to register on the [IAM Handwriting Database website](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) to acces the download links.
    * Once registered, download the following:
        * **lines.tgz**: This contains the images of individual handwritten lines.
        * **ascii.tgz**: This contains the ground truth text transcriptions, including `lines.txt`.

2.  **Extract and Organize:**
    * Extract the `lines.tgz` archive. This will create a `lines` folder containing sub folder of all the line images.
    * Extract the `ascii.tgz` archive. Inside, you will find a file named `lines.txt`.
    * Place the `lines` folder inside the `datasets/` directory of this project.
    * Place the `lines.txt` file directly inside the `datasets/` directory.

Once your dataset is set up like this, you should be able to run the training and testing scripts without any issues.

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

  
## BestModel.pth 
Train Loss: 0.0607
Validation Loss: 0.0573
CER: ~1.6% 
WER: ~6.5% 
![Loss-Over-Epoch](https://github.com/user-attachments/assets/63359afd-ffec-4c65-b45c-f5129937d56f)
![Error-Rate-Over-Epochs](https://github.com/user-attachments/assets/b394f044-6a8b-4126-aa9a-c84a70af3c58)

