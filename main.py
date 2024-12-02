import uvicorn
from fastapi import FastAPI, File, UploadFile

from utils import get_dialog_as_text


app = FastAPI()

@app.post("/asr")
async def asr(file: UploadFile = File(...)) -> dict:
    res = await get_dialog_as_text(file)
    return res

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
