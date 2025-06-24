import os
import glob
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image
import io
import base64
from typing import List
import json

# Import model classes
from encoder_decode_models import EncoderDecoderModel

app = FastAPI(title="4-Digit MNIST Encoder-Decoder API", version="1.0.0")

# Global model variable
model = None
device = None

class PredictionResponse(BaseModel):
    predicted_digits: List[int]
    confidence_scores: List[float]
    success: bool
    message: str

class ImageData(BaseModel):
    image_base64: str

def load_latest_model():
    """Load the latest encoder-decoder model from .data/models"""
    global model, device
    
    models_folder = os.getenv('MODELS_FOLDER', '.data/models')
    
    # Find all encoder-decoder model files
    model_files = glob.glob(os.path.join(models_folder, 'encoder_decoder_4digit_epoch_*.pth'))
    
    if not model_files:
        raise FileNotFoundError("No encoder-decoder model files found")
    
    # Get the latest model (highest epoch number)
    latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncoderDecoderModel().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(latest_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from: {latest_model}")
    print(f"Model epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Model validation accuracy: {checkpoint.get('val_accuracy', 'unknown'):.2f}%")
    
    return latest_model

def preprocess_image(image_data):
    """Preprocess image to match training format"""
    # Convert to grayscale and resize to 56x56
    img = image_data.convert('L').resize((56, 56))
    
    # Convert to tensor and normalize
    img_array = np.array(img) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 56, 56)
    
    return img_tensor.to(device)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_latest_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple HTML interface for testing"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>4-Digit MNIST Encoder-Decoder</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 600px; margin: 0 auto; }
            .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            .result { margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>4-Digit MNIST Encoder-Decoder</h1>
            <p>Upload a 56x56 image containing 4 stacked digits (2x2 grid)</p>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-area">
                    <input type="file" id="imageFile" accept="image/*" required>
                    <br><br>
                    <button type="submit">Predict Digits</button>
                </div>
            </form>
            
            <div id="result" class="result" style="display:none;">
                <h3>Prediction Results:</h3>
                <div id="prediction"></div>
            </div>
        </div>
        
        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const fileInput = document.getElementById('imageFile');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select an image file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('prediction').innerHTML = `
                        <p><strong>Predicted Digits:</strong> ${result.predicted_digits.join(', ')}</p>
                        <p><strong>Confidence Scores:</strong> ${result.confidence_scores.map(s => s.toFixed(3)).join(', ')}</p>
                        <p><strong>Status:</strong> ${result.success ? 'Success' : 'Error'}</p>
                        <p><strong>Message:</strong> ${result.message}</p>
                    `;
                } catch (error) {
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('prediction').innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict 4 digits from uploaded image"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess
        img_tensor = preprocess_image(image)
        
        # Generate prediction
        with torch.no_grad():
            generated = model.generate(img_tensor)
            
            # Get confidence scores by running through decoder
            encoder_output = model.encoder(img_tensor)
            start_tokens = torch.full((1, 1), 10, dtype=torch.long, device=device)
            tgt_input = torch.cat([start_tokens, generated[:, :-1]], dim=1)
            
            logits = model.decoder(tgt_input, encoder_output)
            probs = torch.softmax(logits[:, :, :10], dim=-1)  # Only digits 0-9
            
            # Get confidence for predicted digits
            confidence_scores = []
            for i in range(4):
                digit_prob = probs[0, i, generated[0, i]].item()
                confidence_scores.append(digit_prob)
        
        predicted_digits = generated[0].cpu().numpy().tolist()
        
        return PredictionResponse(
            predicted_digits=predicted_digits,
            confidence_scores=confidence_scores,
            success=True,
            message="Prediction successful"
        )
        
    except Exception as e:
        return PredictionResponse(
            predicted_digits=[],
            confidence_scores=[],
            success=False,
            message=f"Error during prediction: {str(e)}"
        )

@app.post("/predict_base64", response_model=PredictionResponse)
async def predict_base64(data: ImageData):
    """Predict 4 digits from base64 encoded image"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Decode base64 image
        image_data = base64.b64decode(data.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess
        img_tensor = preprocess_image(image)
        
        # Generate prediction
        with torch.no_grad():
            generated = model.generate(img_tensor)
            
            # Get confidence scores
            encoder_output = model.encoder(img_tensor)
            start_tokens = torch.full((1, 1), 10, dtype=torch.long, device=device)
            tgt_input = torch.cat([start_tokens, generated[:, :-1]], dim=1)
            
            logits = model.decoder(tgt_input, encoder_output)
            probs = torch.softmax(logits[:, :, :10], dim=-1)
            
            confidence_scores = []
            for i in range(4):
                digit_prob = probs[0, i, generated[0, i]].item()
                confidence_scores.append(digit_prob)
        
        predicted_digits = generated[0].cpu().numpy().tolist()
        
        return PredictionResponse(
            predicted_digits=predicted_digits,
            confidence_scores=confidence_scores,
            success=True,
            message="Prediction successful"
        )
        
    except Exception as e:
        return PredictionResponse(
            predicted_digits=[],
            confidence_scores=[],
            success=False,
            message=f"Error during prediction: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.get("/model_info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"error": "Model not loaded"}
    
    models_folder = os.getenv('MODELS_FOLDER', '.data/models')
    model_files = glob.glob(os.path.join(models_folder, 'encoder_decoder_4digit_epoch_*.pth'))
    latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    checkpoint = torch.load(latest_model, map_location='cpu')
    
    return {
        "model_path": latest_model,
        "epoch": checkpoint.get('epoch', 'unknown'),
        "train_accuracy": checkpoint.get('train_accuracy', 'unknown'),
        "val_accuracy": checkpoint.get('val_accuracy', 'unknown'),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
