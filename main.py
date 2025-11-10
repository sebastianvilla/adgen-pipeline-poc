from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
from campaign_manager import CampaignManager 
from uuid import uuid4
import json
import os
import traceback
from datetime import datetime

app = FastAPI()

# Pydantic model for the brief (matches our JSON structure)
class AssetSpec(BaseModel):
    creative: Optional[str] = None
    has_logo: bool = False
    has_message: bool = False

class ProductBrief(BaseModel):
    id: str
    regions: List[str]
    audience: str
    message: str           # per-product English message
    theme: str
    aspect_ratios: List[str]
    assets: AssetSpec

class CampaignBrief(BaseModel):
    id: str
    products: List[ProductBrief]

# Simple HTML form for input (can be submitted via browser or curl)
with open("ui/index_simple.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# JSON brief template
with open("ui/brief_template.json", "r", encoding="utf-8") as f:
    brief_template = f.read()

html_content = html_content.replace("@@brief_template@@", brief_template)

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("ui/index.html", "r", encoding="utf-8") as f:
        html_ui = f.read()
    return HTMLResponse(content=html_ui)

@app.get("/simple", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=html_content)

def _run_job(job_id: str, brief_dict: dict):
    try:
        manager = CampaignManager(job_id)
        manager.process(brief_dict)
    except Exception as e:
        outdir = os.path.join("outputs", job_id)
        os.makedirs(outdir, exist_ok=True)

        # Emergency log.txt so you’re never left without a log
        log_path = os.path.join(outdir, "log.txt")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()}Z - ERROR - Unhandled exception in _run_job: {e}\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
            f.write("\n")

        # Emergency report.json so /jobs/<id> has something to read
        report_path = os.path.join(outdir, "report.json")
        emergency = {
            "job_id": job_id,
            "campaign_id": brief_dict.get("id", "unknown"),
            "status": "failed",
            "error": str(e),
            "progress": {"last_updated": f"{datetime.utcnow().isoformat()}Z"}
        }
        # write atomically to avoid partial JSON
        tmp = f"{report_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(emergency, f, indent=4)
        os.replace(tmp, report_path)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.post("/process_brief")
async def process_brief(request: Request, background_tasks: BackgroundTasks): 
    try:
        body = await request.body()
        brief_data = json.loads(body)
        brief = CampaignBrief(**brief_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    # job_id = str(uuid4())
    job_id = brief_data.get("id", str(uuid4()))
    # manager = CampaignManager(job_id)
    # result = manager.process(brief.dict())  # Pass dict to manager

    # Run in background
    background_tasks.add_task(_run_job, job_id, brief.dict())

    return {
        "job_id": job_id,
        "status": "queued",
        "report_path": f"outputs/{job_id}/report.json"
    }

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Return the current report.json for a job.
    - 200: returns parsed JSON report
    - 404: job/report not found
    - 503: report file exists but is mid-write (invalid JSON)
    """
    report_path = os.path.join("outputs", job_id, "report.json")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found for this job_id.")

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=503, detail="Report is being updated; please retry.")
    return JSONResponse(content=data)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)