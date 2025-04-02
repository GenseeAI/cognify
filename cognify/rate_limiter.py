from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading, queue, time, uuid
from litellm import completion, RateLimitError
from dataclasses import dataclass
import litellm
import sys

# litellm.set_verbose=True

app = FastAPI()

# --- Configuration ---
DEFAULT_RATE = 500       # Maximum calls per second, used when response header has no rate limit info
                         # equivalent to Tier-5 openai-4o-mini rate limit
NUM_WORKERS = 256        # Number of worker threads per model

# Holds (job_id, task data) per model
task_queue_pool = {}
# Maps job_id -> result dict
job_results = {}             
# Maps job_id -> threading.Event
job_events = {}              
rate_limit_pool = {}
rate_semaphore_pool = {}

@dataclass
class CompletionRequest:
    model: str
    messages: list
    model_kwargs: dict

# --- Rate Limiter Thread ---
def rate_limiter(semaphore, name):
    while True:
        time.sleep(1.0 / rate_limit_pool[name])
        semaphore.release()

# --- Worker Thread Function ---
def worker(semaphore, task_queue):
    while True:
        job_id, req = task_queue.get()
        # Wait for a token (rate limiting)
        semaphore.acquire()
        try:
            # Call the underlying completion function.
            response = completion(req.model, req.messages, **req.model_kwargs)
            result = {"result": {**response.model_dump(), "_hidden_params": response._hidden_params, "_response_headers": response._response_headers}}
            job_results[job_id] = result
            # increase rate limit by 1
            rate_limit_pool[req.model] += 1
        except RateLimitError as e:
            # reduce rate limit by half and put to the back of the queue
            rate_limit_pool[req.model] /= 2
            task_queue.put((job_id, req))
        except Exception as e:
            job_results[job_id] = {"error": str(e)}
        # Signal that the job is done.
        if job_id in job_results:
            job_events[job_id].set()
        task_queue.task_done()

def first_time_request(job_id, req: CompletionRequest):
    try:
        response = completion(req.model, req.messages, **req.model_kwargs)
        result = {"result": {**response.model_dump(), "_hidden_params": response._hidden_params, "_response_headers": response._response_headers}}
        job_results[job_id] = result
    except Exception as e:
        job_results[job_id] = {"error": str(e)}
        raise e
    # Signal that the job is done.
    if job_id in job_results:
        job_events[job_id].set()
        
    # setup rate limit for this model
    if limit := response._response_headers.get("x-ratelimit-remaining-requests", None):
        rate = (int(limit) + 1) / 60 # to account for the current request
        # print(f"Rate limit for {req.model}: {rate}")
    else:
        rate = DEFAULT_RATE
    # start workers
    for _ in range(NUM_WORKERS):
        t = threading.Thread(target=worker, args=(
            rate_semaphore_pool[req.model],
            task_queue_pool[req.model]
        ), daemon=True)
        t.start()
    # start ticket generator
    rate_limit_pool[req.model] = rate
    threading.Thread(target=rate_limiter, args=(
        rate_semaphore_pool[req.model],
        req.model
    ), daemon=True).start()
        

# --- FastAPI Endpoint ---
@app.post("/completion_endpoint")
def completion_endpoint(req: CompletionRequest):

    # Create a unique job ID and an Event to wait for the result.
    job_id = str(uuid.uuid4())
    event = threading.Event()
    job_events[job_id] = event

    # Enqueue the task.
    # If model is new, create a new limiter for it
    if req.model not in task_queue_pool:
        task_queue_pool[req.model] = queue.Queue()
        rate_semaphore_pool[req.model] = threading.Semaphore(0)
        first_time_request(job_id, req)
    else:
        task_queue_pool[req.model].put((job_id, req))
    
    # Wait for the worker to process the task 
    event.wait()
    # if not event.wait(timeout=30):
    #     job_events.pop(job_id, None)
    #     job_results.pop(job_id, None)
    #     raise HTTPException(status_code=504, detail="Task timed out")
    
    result = job_results.pop(job_id, None)
    job_events.pop(job_id, None)
    if result is None:
        raise HTTPException(status_code=500, detail="Job processing error")
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

import uvicorn

def run_rate_limiter(port):
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

if __name__ == "__main__":
    run_rate_limiter(int(sys.argv[1]))