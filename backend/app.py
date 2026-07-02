from pydoc import text

from fastapi import FastAPI
from fastapi import  Request
from fastapi.templating import Jinja2Templates

from applicatin import model

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )
     
app.get("/predict")
