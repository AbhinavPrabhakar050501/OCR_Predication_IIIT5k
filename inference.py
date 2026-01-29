"""
Inference script for CRNN OCR model.

- Uses PyTorch CPU-optimized model (crnn_robust_best.pth)
- Supports variable-width text images
- ONNX model is provided for benchmarking only

"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import string

# ============================================
# CHARACTER MAPPING
# ============================================

CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase
char2idx = {c: i + 1 for i, c in enumerate(CHARS)}
idx2char = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(char2idx) + 1


# ============================================
# MODEL ARCHITECTURE
# ============================================

class ImprovedCRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            
            nn.Conv2d(512, hidden_size, 2, 1, 0),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
        )
        
        self.rnn = nn.LSTM(
            hidden_size, hidden_size, 2,
            bidirectional=True, batch_first=True, dropout=0
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        features = self.cnn(x)
        features = features.squeeze(2).permute(0, 2, 1)
        seq, _ = self.rnn(features)
        return self.fc(seq)


# ============================================
# INFERENCE ENGINE
# ============================================

class OCREngine:
    def __init__(self, model_path="crnn_robust_best.pth"):
        """Initialize PyTorch model."""
        print("Loading model...")
        self.device = torch.device('cpu')
        
        self.model = ImprovedCRNN(NUM_CLASSES)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
        self.model.to(self.device)
        
        # Disable dropout for inference
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0
        
        print("Model loaded successfully!")
    
    def preprocess(self, image, img_height=32):
        """Preprocess image for inference."""
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize maintaining aspect ratio
        h, w = image.shape
        new_w = max(1, int(w * (img_height / h)))
        image = cv2.resize(image, (new_w, img_height))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        
        # Convert to tensor
        image = torch.tensor(image).unsqueeze(0).unsqueeze(0)
        
        return image
    
    def decode(self, logits):
        """CTC greedy decoder."""
        preds = logits.argmax(2).cpu().numpy()
        
        decoded = []
        prev = None
        
        for p in preds[0]:
            if p == 0:
                prev = None
                continue
            if p != prev:
                decoded.append(idx2char.get(p, '?'))
                prev = p
        
        return "".join(decoded)
    
    def predict(self, image):
        """Run inference."""
        # Preprocess
        img = self.preprocess(image)
        img = img.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(img)
        
        # Decode
        text = self.decode(logits)
        
        return text


# ============================================
# INITIALIZE ENGINE
# ============================================

try:
    engine = OCREngine()
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False


# ============================================
# GRADIO INTERFACE
# ============================================

def predict_text(image):
    """
    Prediction function for Gradio.
    """
    if not model_loaded:
        return "❌ Model not loaded. Please check if crnn_robust_best.pth exists."
    
    if image is None:
        return "⚠️ Please upload an image"
    
    try:
        # Get prediction
        predicted_text = engine.predict(image)
        
        # Format output
        if len(predicted_text) == 0:
            return "⚠️ No text detected"
        
        return f"📝 **{predicted_text}**"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks(
    title="OCR Text Recognition",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple")
) as demo:
    
    gr.Markdown("""
    # OCR Text Recognition

    Deep Learning OCR model built with CRNN architecture for text recognition.
    
    ### How to use
    1. Upload an image containing text
    2. Click "Recognize Text"
    3. View the predicted text
    4. For best results use Clear, well-lit images.

    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="📤 Upload Text Image")
            submit_btn = gr.Button("🚀 Recognize Text", variant="primary", size="lg")
        
        with gr.Column():
            output_text = gr.Markdown(label="Predicted Text", value="Upload an image to start...")
    
    submit_btn.click(
        fn=predict_text,
        inputs=input_image,
        outputs=output_text
    )
    
    


# ============================================
# LAUNCH
# ============================================

if __name__ == "__main__":
    demo.launch()


