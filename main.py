from backend.app import app


if __name__ == "__main__":

    import uvicorn

    print("=" * 80)
    print("STARTING TELECOM AGENTIC FAULT MANAGEMENT SYSTEM")
    print("=" * 80)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )