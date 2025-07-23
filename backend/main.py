from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend import routes

app = FastAPI(title="DeFakeIt Fake News Detection API")

# Allow React frontend (adjust origin in production)
origins = [
    "http://localhost:3000",  # React dev server
    "http://127.0.0.1:3000",
    # Add your deployed frontend URL here later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods: GET, POST, etc.
    allow_headers=["*"],  # Allow all headers
)

# Include API routes
app.include_router(routes.router)

@app.get("/")
async def root():
    return {"message": "Welcome to DeFakeIt API"}
