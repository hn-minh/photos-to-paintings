import gradio as gr
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/predict"

STYLE_IMAGES = {
    "cezanne": "data/style_imgs/cezanne.jpg",
    "monet": "data/style_imgs/monet.jpg",
    "vangogh": "data/style_imgs/vangogh.jpg"
}

def update_style_reference(style_name):
    return STYLE_IMAGES.get(style_name)

def process_image(input_image, style_name):
    if input_image is None:
        raise gr.Error("Vui lòng tải lên một bức ảnh!")
    
    buf = io.BytesIO()
    input_image.save(buf, format="JPEG")
    buf.seek(0)
    
    try:
        response = requests.post(
            API_URL,
            data={"style": style_name},
            files={"file": ("image.jpg", buf, "image/jpeg")}
        )
        
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            raise gr.Error(f"Lỗi từ server: {response.json().get('detail', 'Unknown error')}")
            
    except requests.exceptions.ConnectionError:
        raise gr.Error("Không thể kết nối tới Backend. Hãy chắc chắn FastAPI đang chạy!")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Style Transfer
        Tải ảnh của bạn lên và chọn phong cách hội họa yêu thích. Hệ thống sẽ chuyển đổi bức ảnh của bạn thành một tác phẩm nghệ thuật!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Ảnh của bạn")
            style_dropdown = gr.Dropdown(
                choices=["cezanne", "monet", "vangogh"], 
                value="vangogh", 
                label="Chọn Phong cách (Style)"
            )
            style_ref_img = gr.Image(
                value=STYLE_IMAGES["vangogh"], 
                label="Ảnh phong cách gốc trong Dataset", 
                interactive=False
            )
            submit_btn = gr.Button("Chuyển đổi", variant="primary")
            
        with gr.Column(scale=1):
            output_img = gr.Image(label="Kết quả")
            
    style_dropdown.change(
        fn=update_style_reference, 
        inputs=style_dropdown, 
        outputs=style_ref_img
    )
    
    submit_btn.click(
        fn=process_image, 
        inputs=[input_img, style_dropdown], 
        outputs=output_img
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)