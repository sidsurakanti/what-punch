from fastapi import FastAPI, WebSocket

app = FastAPI()


@app.get("/")
def root():
    return {"status": "OK"}


@app.websocket("/predict")
async def predict(websocket: WebSocket):
    await websocket.accept()
    while 1:
        data = await websocket.receive_bytes()
        print(data)
        await websocket.send_json({"recieved": True})
