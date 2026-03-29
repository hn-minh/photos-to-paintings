import io
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from models import TransformerNet
from utils import process_image, tensor_to_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {}
style_names = ["cezanne", "monet", "vangogh"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    for style in style_names:
        model = TransformerNet().to(device)
        try:
            model.load_state_dict(torch.load(f"output/checkpoints/{style}.pth", map_location=device))
            model.eval()
            models[style] = model
            print(f"✅ Đã load thành công model: {style}")
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file checkpoints/{style}.pth")
    
    yield
    
    models.clear()

app = FastAPI(title="Style Transfer API", version="1.0", lifespan=lifespan)

@app.post("/predict")
async def predict(style: str = Form(...), file: UploadFile = File(...)):
    if style not in models:
        raise HTTPException(
            status_code=400, 
            detail=f"Style '{style}' không hợp lệ. Hãy chọn: {', '.join(style_names)}"
        )
    
    try:
        image_bytes = await file.read()
        
        input_tensor = process_image(image_bytes, device)
        
        with torch.no_grad():
            output_tensor = models[style](input_tensor)
            
        result_image = tensor_to_image(output_tensor)
        
        buf = io.BytesIO()
        result_image.save(buf, format="JPEG")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")