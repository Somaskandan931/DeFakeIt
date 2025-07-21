from fastapi import FastAPI
from backend import routes

app = FastAPI(title="DeFakeIt Fake News Detection API")

# Include API routes
app.include_router(routes.router)

@app.get("/")
async def root():
    return {"message": "Welcome to DeFakeIt API"}
