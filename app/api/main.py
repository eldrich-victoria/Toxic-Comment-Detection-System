from fastapi import FastAPI
from app.api.routes import router
from app.api.middleware import RequestTimingMiddleware
from app.api.exceptions import global_exception_handler

app = FastAPI(title="Toxic Comment Benchmark API", version="1.0.0")

app.add_middleware(RequestTimingMiddleware)
app.add_exception_handler(Exception, global_exception_handler)

app.include_router(router)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
