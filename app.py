import gradio as gr
import easyocr
import cv2
import numpy as np
from PIL import Image

# Create an EasyOCR Reader
reader = easyocr.Reader(['en'])

def process_image(image):
    # Convert the PIL image to a numpy array (compatible with OpenCV)
    image_np = np.array(image)

    # Convert the image to RGB (OpenCV loads as BGR, EasyOCR expects RGB)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Use EasyOCR to read text from the image
    result = reader.readtext(image_rgb)

    # Draw bounding boxes around detected text
    for (bbox, text, prob) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(image_np, top_left, bottom_right, (0, 255, 0), 2)
    
    # Convert back to RGB for display
    result_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    
    # Combine detected text and their confidence scores
    detected_text = "\n".join([f"Detected text: {text}, Confidence: {prob:.2f}" for (_, text, prob) in result])
    
    return result_image, detected_text

# Gradio Interface
interface = gr.Interface(
    fn=process_image, 
    inputs="image", 
    outputs=["image", "text"], 
    title="OCR with EasyOCR", 
    description="Upload an image, and the system will detect text using EasyOCR and display it."
)

# Launch the interface
interface.launch(share=True)
