from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import torch
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
async def health_check() -> Dict[str, str]:
    return {
        "status": "healthy",
        "service": "LexAI",
        "version": "0.1.0"
    }


@router.get("/gpu")
async def gpu_status() -> Dict[str, Any]:
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
            
            return {
                "cuda_available": True,
                "device_count": device_count,
                "current_device": current_device,
                "device_name": device_name,
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_reserved_gb": round(memory_reserved, 2),
                "cuda_version": torch.version.cuda
            }
        else:
            return {
                "cuda_available": False,
                "message": "CUDA is not available"
            }
    except Exception as e:
        logger.error(f"Error checking GPU status: {e}")
        raise HTTPException(status_code=500, detail=str(e))