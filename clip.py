from PIL import Image
import torch
import open_clip

def clip_predictions(img: Image.Image, categories: list[str]):
    # Load model and tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu')
    tokenizer = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")

    # Load and preprocess image
    img = img.convert("RGB")
    img_p = preprocess(img).unsqueeze(0)

    # Tokenize text
    text = tokenizer(categories)

    # Encode image and text
    with torch.no_grad(), torch.autocast("mps"):
        image_features = model.encode_image(img_p)
        text_features = model.encode_text(text)

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarity and probabilities
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Format and return
    predictions = text_probs.squeeze().cpu() * 100
    predicted_label = categories[predictions.argmax()]

    return predictions.tolist(), predicted_label

from PIL import Image
import torch
import open_clip

# Only initialize these once globally
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer("ViT-B-32")

def estimate_boat_scale(image: Image.Image) -> str:
    """Estimate the plausible visual size (small, medium, large) a boat should have in the image."""
    boat_size_prompts = {
        "super small": "a very small boat far away on the sea",
        "small": "a small boat far away on the sea",
        "medium": "a medium-sized boat in the middle distance",
        "large": "a large boat close to the camera"
    }

    image = image.convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0)

    text_descriptions = list(boat_size_prompts.values())
    labels = list(boat_size_prompts.keys())

    tokens = tokenizer(text_descriptions)

    with torch.no_grad(), torch.autocast("mps"):
        img_features = model.encode_image(img_tensor)
        text_features = model.encode_text(tokens)

        img_features /= img_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        probs = (100.0 * img_features @ text_features.T).softmax(dim=-1)

    predicted_idx = probs.squeeze().argmax().item()
    return labels[predicted_idx]

print(clip_predictions(Image.open("realistic.png"), ["realistic boat size", "unrealistic boat size"]))
