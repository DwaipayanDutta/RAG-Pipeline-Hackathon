import time
import uuid
import logging
from typing import Dict, Any

logger = logging.getLogger("rag_pipeline")

class RequestContext:
    def __init__(self):
        self.request_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.stages = {}

    def stage_start(self, stage: str):
        self.stages[stage] = {"start": time.time()}

    def stage_end(self, stage: str):
        if stage in self.stages:
            self.stages[stage]["end"] = time.time()
            self.stages[stage]["duration_ms"] = (self.stages[stage]["end"] - self.stages[stage]["start"]) * 1000

    def log(self, extra: Dict[str, Any] = None):
        total_duration = (time.time() - self.start_time) * 1000
        log_data = {
            "request_id": self.request_id,
            "total_duration_ms": total_duration,
            "stages": self.stages,
            **(extra or {})
        }
        logger.info("Request completed", extra=log_data)