import os
import argparse
import torch.nn as nn
from src.HTR_Model import ModelHTR, create_dataloader
from src.visualization import plot_text_lengths, plot_char_distribution

def main():
    parser = argparse.ArgumentParser(description="Line-level HTR CLI")
    parser.add_argument("--mode", choices=["train", "predict"], help="Mode: train or predict")
    parser.add_argument("--model", type=str, help="Path to load model checkpoint")
    parser.add_argument("--save_path", type=str, help="Path to save trained model")
    parser.add_argument("--image_path", type=str, help="Path to input image (for prediction)")
    parser.add_argument("--epochs", type=int, default=75, help="Number of training epochs")
    parser.add_argument("--visualize", action="store_true", help="Enable dataset visualizations")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model on the validation set")

    args = parser.parse_args()
    
    # Require at least one action
    if not args.mode and not args.eval:
        parser.error("You must provide either --mode or --eval.")
    
    model, args = initialize_model(args)

    if args.mode == "train":
        model.train(epochs=args.epochs, save_path=args.save_path)
    elif args.mode == "predict":
        run_prediction(model, args)

    # Visualize data
    if args.visualize:
        dataset = model.train_dataset.dataset
        plot_text_lengths(dataset)
        plot_char_distribution(dataset)
    
    # Evaluate model
    if args.eval:
        evaluate(model)
        
def initialize_model(args, default_path="best_model.pth"):
    model = ModelHTR()
    
    # Load model if file exists
    model_path = args.model if args.model else default_path
    if os.path.exists(model_path):
        model.load(model_path)
    else:
        print(f"Warning: model checkpoint not found at '{model_path}'. Initializing new model.")
    
    # Set save_path to model path if not provided
    if not args.save_path:
        args.save_path = model_path
    
    return model, args

def run_prediction(model, args):
    # Ensure that there is an image path
    if not args.image_path:
        raise ValueError("You must provide --image_path in predict mode.")

    # If the path is dir, predict all images in it
    if os.path.isdir(args.image_path):
        image_files = [f for f in os.listdir(args.image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            raise ValueError("No image files found in directory.")
        print("Batch predicting...")
        
        for fname in sorted(image_files):
            full_path = os.path.join(args.image_path, fname)
            pred = model.predict(full_path, visualize=args.visualize)
            print(f"--{fname}: {pred}")
    
    # Predict single image
    else:
        print("Predicting...")
        pred = model.predict(args.image_path, visualize=args.visualize)
        print(f"Predicted text: {pred}")
    
def evaluate(model):
    val_loader = create_dataloader(model.val_dataset, shuffle=False)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    val_loss, cer, wer = model.evaluate(val_loader, criterion)
    print(f"Validation Loss: {val_loss:.4f}, CER: {cer:.4f}, WER: {wer:.4f}")
    
if __name__ == "__main__":
    main()