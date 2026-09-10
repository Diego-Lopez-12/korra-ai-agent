"""
hazard_logger.py

Logging helpers for hazard events.
"""

def structural_hazard(worker_name: str, request_id: str, blocking_request_id: str):
    print(f"[STRUCTURAL HAZARD] Worker {worker_name} busy. Request {request_id} waiting (blocked by Request {blocking_request_id})")

def forwarding(producer_name: str, consumer_name: str, request_id: str):
    print(f"[FORWARDING] Request {request_id}: {producer_name} -> {consumer_name}")

def stall(worker_name: str, request_id: str):
    print(f"[STALL] Request {request_id} stalled waiting on {worker_name}")

def bubble_cycle(cycle_num: int):
    print(f"[BUBBLE CYCLE {cycle_num}]")

def control_hazard(request_id: str, reason: str):
    print(f"[CONTROL HAZARD] Request {request_id}: {reason}")

def flush(request_id: str, workers: list[str]):
    print(f"[FLUSH] Request {request_id}: discarding queued workers {workers}")

def resume(worker_name: str, request_id: str):
    print(f"[RESUME] Request {request_id} now running on {worker_name}")