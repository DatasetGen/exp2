import ollama
from PIL import Image
import io

def answer_question_about_image(image: Image.Image, question: str, model: str = "llava:latest") -> str:
    """
    Use the LLaVA model via Ollama to answer a question about a Pillow image.

    Args:
        image (PIL.Image.Image): The image to analyze.
        question (str): The question to ask about the image.
        model (str): The model to use (default is 'llava:latest').

    Returns:
        str: The model's response.
    """
    try:
        # Convert Pillow image to bytes (in PNG format)
        img_bytes_io = io.BytesIO()
        image.save(img_bytes_io, format='PNG')
        image_bytes = img_bytes_io.getvalue()

        # Send image and question to Ollama
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": question}
            ],
            images=[image_bytes]
        )

        return response['message']['content']

    except Exception as e:
        return f"Error: {str(e)}"

def answer_question(question: str, model: str = "llama3.2") -> str:
    """
    Use the LLaVA model via Ollama to answer a question about a Pillow image.

    Args:
        image (PIL.Image.Image): The image to analyze.
        question (str): The question to ask about the image.
        model (str): The model to use (default is 'llava:latest').

    Returns:
        str: The model's response.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": question}
            ],
        )

        return response['message']['content']

    except Exception as e:
        return f"Error: {str(e)}"

