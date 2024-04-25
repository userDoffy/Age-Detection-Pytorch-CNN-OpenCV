from fastapi import FastAPI,UploadFile,File,Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn 
import numpy as np
from predict import predict_age
import cv2
from PIL import Image
from io import BytesIO
import base64

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post('/predict')
async def predict(file:UploadFile=File()):
    img_file=await file.read()
    img_arr=np.array(Image.open(BytesIO(img_file)))
    image=cv2.cvtColor(img_arr,cv2.IMREAD_COLOR)
    prediction=predict_age(image)
    print(prediction)
    return {
        "age":prediction
    }

if __name__=="__main__":
    uvicorn.run(app,port=8000,host='localhost')
