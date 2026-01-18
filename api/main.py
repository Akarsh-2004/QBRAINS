#!/usr/bin/env python3
"""
REST API for Quantum Emotion Pipeline
FastAPI-based web service
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any
import sys
from pathlib import Path
import numpy as np
import io
import base64

sys.path.append(str(Path(__file__).parent.parent))

from src.quantum_pipeline_integrated import IntegratedQuantumPipeline
from src.realtime_processor import RealTimeProcessor

app = FastAPI(title="Quantum Emotion Pipeline API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = IntegratedQuantumPipeline()
realtime_processor = None

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# Request models
class TextRequest(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


class EEGDataRequest(BaseModel):
    eeg_data: list  # List of lists (channels x time_steps)
    sample_rate: Optional[int] = 256


# Routes
@app.get("/")
async def root():
    """Root endpoint - serve dashboard"""
    dashboard_path = Path(__file__).parent / "static" / "index.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    else:
        return {
            "name": "Quantum Emotion Pipeline API",
            "version": "1.0.0",
            "endpoints": {
                "/process/text": "POST - Process text input",
                "/process/audio": "POST - Process audio file",
                "/process/video": "POST - Process video file",
                "/process/eeg": "POST - Process EEG data",
                "/chat": "POST - Chat interface",
                "/realtime/start": "POST - Start real-time processing",
                "/realtime/stop": "POST - Stop real-time processing",
                "/realtime/result": "GET - Get latest real-time result",
                "/memory": "GET - Get memory summary",
                "/health": "GET - Health check"
            }
        }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "pipeline": "ready"}


@app.post("/process/text")
async def process_text(request: TextRequest):
    """Process text input"""
    try:
        result = pipeline.process(text=request.text, context=request.context)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/audio")
async def process_audio(file: UploadFile = File(...)):
    """Process audio file"""
    try:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        result = pipeline.process(audio_path=tmp_path)
        
        # Cleanup
        import os
        os.unlink(tmp_path)
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/video")
async def process_video(file: UploadFile = File(...)):
    """Process video file"""
    try:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        result = pipeline.process(video_path=tmp_path)
        
        # Cleanup
        import os
        os.unlink(tmp_path)
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/eeg")
async def process_eeg(request: EEGDataRequest):
    """Process EEG data"""
    try:
        eeg_array = np.array(request.eeg_data)
        result = pipeline.process(eeg_data=eeg_array)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/eeg_file")
async def process_eeg_file(file: UploadFile = File(...)):
    """Process EEG CSV file"""
    try:
        import tempfile
        import pandas as pd
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        result = pipeline.process(eeg_path=tmp_path)
        
        # Cleanup
        import os
        os.unlink(tmp_path)
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat interface - automatically updates memory"""
    try:
        # Process through pipeline to get full result with emotions
        result = pipeline.process(text=request.message, context=request.context)
        # Extract formatted response
        response_text = result.get('final_output', {}).get('formatted_text', '')
        if not response_text:
            response_text = result.get('quantum_superposition', {}).get('collapsed_emotion', 'Processed')
        
        # Return response with emotion data for UI
        return {
            "response": response_text,
            "quantum_superposition": result.get('quantum_superposition'),
            "emotions": result.get('raw_emotions', {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/file")
async def process_file(file: UploadFile = File(...)):
    """Generic file processing endpoint - auto-detects file type"""
    try:
        import tempfile
        import os
        
        # Determine file type from content type or extension
        content_type = file.content_type or ""
        filename = file.filename or ""
        
        # Get appropriate suffix
        if content_type.startswith("audio/") or filename.endswith((".wav", ".mp3", ".m4a", ".flac")):
            suffix = ".wav"
            endpoint_type = "audio"
        elif content_type.startswith("video/") or filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
            suffix = ".mp4"
            endpoint_type = "video"
        elif filename.endswith(".csv"):
            suffix = ".csv"
            endpoint_type = "eeg"
        else:
            suffix = ".tmp"
            endpoint_type = "unknown"
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process based on type
        if endpoint_type == "audio":
            result = pipeline.process(audio_path=tmp_path)
        elif endpoint_type == "video":
            result = pipeline.process(video_path=tmp_path)
        elif endpoint_type == "eeg":
            result = pipeline.process(eeg_path=tmp_path)
        else:
            # Try as text file
            try:
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                result = pipeline.process(text=text_content)
            except:
                raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Cleanup
        os.unlink(tmp_path)
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/realtime/start")
async def start_realtime(camera_id: int = 0, mode: str = "local"):
    """
    Start real-time processing
    mode: "local" (uses server camera/mic) or "stream" (uses WebSocket injection)
    """
    global realtime_processor
    try:
        if realtime_processor is None:
            realtime_processor = RealTimeProcessor(pipeline=pipeline)
        
        if mode == "local":
            realtime_processor.start_video_stream(camera_id=camera_id)
            realtime_processor.start_audio_stream()
        else:
            # For stream mode, we just need the processing worker running
            realtime_processor.running = True
            if not realtime_processor.processing_thread or not realtime_processor.processing_thread.is_alive():
                import threading
                realtime_processor.processing_thread = threading.Thread(
                    target=realtime_processor._process_streams_worker,
                    daemon=True
                )
                realtime_processor.processing_thread.start()
        
        return {"status": "started", "camera_id": camera_id, "mode": mode}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Bidirectional streaming endpoint for real-time video/audio analysis"""
    global realtime_processor
    await websocket.accept()
    
    if realtime_processor is None:
        realtime_processor = RealTimeProcessor(pipeline=pipeline)
        realtime_processor.running = True
        import threading
        realtime_processor.processing_thread = threading.Thread(
            target=realtime_processor._process_streams_worker,
            daemon=True
        )
        realtime_processor.processing_thread.start()

    try:
        while True:
            # Receive data (can be text/JSON or binary)
            data = await websocket.receive()
            
            if "text" in data:
                # Handle JSON metadata or base64 frames
                import json
                try:
                    msg = json.loads(data["text"])
                    if msg.get("type") == "frame":
                        realtime_processor.process_frame_binary(msg["data"])
                    elif msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except:
                    pass
            
            elif "bytes" in data:
                # Handle binary audio chunks
                realtime_processor.process_audio_binary(data["bytes"])
            
            # Periodically broadcast latest result
            result = realtime_processor.get_latest_result(timeout=0.01)
            if result:
                await websocket.send_json({
                    "type": "result",
                    "data": result
                })
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Don't stop engine automatically, let the user decide via /stop
        pass


@app.post("/realtime/stop")
async def stop_realtime():
    """Stop real-time processing"""
    global realtime_processor
    try:
        if realtime_processor:
            realtime_processor.stop()
            realtime_processor = None
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/realtime/result")
async def get_realtime_result():
    """Get latest real-time processing result"""
    global realtime_processor
    try:
        if realtime_processor:
            result = realtime_processor.get_latest_result(timeout=0.5)
            if result:
                return JSONResponse(content=result)
            else:
                return {"status": "no_result"}
        else:
            return {"status": "not_running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory")
async def get_memory():
    """Get memory summary"""
    try:
        summary = pipeline.get_memory_summary()
        return JSONResponse(content=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory")
async def clear_memory():
    """Clear memory"""
    try:
        pipeline.clear_memory()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

