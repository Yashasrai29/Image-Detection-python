import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from io import BytesIO
from fastapi.responses import FileResponse

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained YOLOv5 model (face detection)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='face_yolov5.pt', force_reload=True)

# Route for detecting faces
@app.post("/detect-face/")
async def detect_face(file: UploadFile = File(...)):
    # Read the image file from the request
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))
    
    # Convert to numpy array and perform detection
    img_array = np.array(image)
    
    # Use the YOLO model to detect faces
    results = model(img_array)


    # Filter detections to get only faces (class 0 in YOLOv5 is person, and we assume faces are detected as 'person')
    faces = [det for det in results.xywh[0] if det[5] == 0]  # Class 0: person (human face detection)

    # Draw bounding boxes around detected faces
    for face in faces:
        x1, y1, x2, y2 = map(int, face[:4])
        img_array = cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the image with faces highlighted
    output_image_path = "output_image_with_faces.jpg"
    cv2.imwrite(output_image_path, img_array)

    # Optionally, you can store the image in a database or filesystem for future reference

    return FileResponse(output_image_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
