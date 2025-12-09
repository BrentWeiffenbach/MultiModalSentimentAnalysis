import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import textwrap
import numpy as np

from src.multimodal_sentiment_analysis import MultiModalSentimentAnalysis


def load_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

    # Transform must match training (Resize + ToTensor)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, tensor


def load_text(text_input, tokenizer):
    # Check if text_input is a file path
    if os.path.exists(text_input) and os.path.isfile(text_input):
        with open(text_input, "r") as f:
            text = f.read().strip()
    else:
        text = text_input

    encodings = tokenizer(
        [text], return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    return text, encodings


def visualize_result(image, text, probabilities, prediction_idx, output_path):
    classes = ["Negative", "Neutral", "Positive"]
    predicted_class = classes[prediction_idx]
    
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    
    # Image subplot (Left side)
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title("Input Image")
    
    # Text info subplot (Top Right)
    ax2 = plt.subplot(2, 2, 2)
    ax2.axis('off')
    
    # Wrap text
    wrapped_text = textwrap.fill(text, width=40)
    
    info_text = (
        f"Input Text:\n{wrapped_text}\n\n"
        f"Prediction: {predicted_class}\n\n"
        f"Probabilities:\n"
        f"Negative: {probabilities[0]:.4f}\n"
        f"Neutral:  {probabilities[1]:.4f}\n"
        f"Positive: {probabilities[2]:.4f}"
    )
    
    ax2.text(0.05, 0.5, info_text, fontsize=11, va='center', fontfamily='monospace')
    
    # Bar chart for probabilities (Bottom Right)
    ax3 = plt.subplot(2, 2, 4)
    bars = ax3.bar(classes, probabilities, color=["red", "gray", "green"])
    ax3.set_ylim(0, 1.1) # Little extra space for labels
    ax3.set_title("Confidence Scores")
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # Remove top and right spines for cleaner look
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Result saved to {output_path}")
    plt.close()
def main():
    parser = argparse.ArgumentParser(
        description="Inference for Multi-Modal Sentiment Analysis"
    )
    parser.add_argument(
        "--image-path", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--text-input",
        type=str,
        required=True,
        help="Input text string or path to text file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="inference_result.png",
        help="Path to save output image",
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Use HuggingFace architecture (must match training)",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load Inputs
    print(f"Loading image from {args.image_path}...")
    pil_image, image_tensor = load_image(args.image_path)
    if pil_image is None:
        return

    print("Processing text...")
    text_content, text_encodings = load_text(args.text_input, tokenizer)

    # Move inputs to device
    image_tensor = image_tensor.to(device)
    input_ids = text_encodings["input_ids"].to(device)
    attention_mask = text_encodings["attention_mask"].to(device)

    # Initialize Model
    print("Initializing model...")
    model = MultiModalSentimentAnalysis(
        num_classes=3,
        dropout=0.1,  # Doesn't matter for inference
        freeze_encoders=False,  # Doesn't matter for inference
        pretrained_vit_path=None,  # Not needed as we load full state dict
        pretrained_bert_path=None,  # Not needed as we load full state dict
        use_huggingface=args.use_huggingface,
        use_fusion=True
    )

    # Load Weights
    print(f"Loading weights from {args.model_path}...")
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.to(device)
    model.eval()

    # Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(image_tensor, input_ids, attention_mask)
        probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
        prediction_idx = np.argmax(probabilities)

    # Visualize
    visualize_result(
        pil_image, text_content, probabilities, prediction_idx, args.output_path
    )


if __name__ == "__main__":
    main()
