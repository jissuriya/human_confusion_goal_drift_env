from fastapi import FastAPI
import os
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    return {"message": "environment reset"}

@app.post("/step")
def step():
    return {
        "observation": "dummy",
        "reward": 0,
        "done": False,
        "info": {}
    }

def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host=host, port=port)

if _name_ == "_main_":
    main()
