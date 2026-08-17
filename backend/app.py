from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.login_routes import router as login_router
from backend.routes.incident_routes import router as incident_router


app = FastAPI(
    title="Telecom Agentic Fault Management System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(login_router)
app.include_router(incident_router)


@app.get("/")
def root():
    return {
        "service": "Telecom Agentic Fault Management",
        "status": "online"
    }
    