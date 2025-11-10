# Adgen Pipeline

## Overview
This proof of concept demonstrates an end-to-end creative generation pipeline for multi-region ad campaigns.  It automates creative expansion, localization, brand compliance, and asset composition through a modular, worker-based architecture.

The pipeline accepts a structured campaign brief and produces final ad variants across multiple aspect ratios and locales, along with legal and brand checks, logging and a job report.

When creative assets are not provided, it uses brand-defined product templates and generates backgrounds for the ads. This approach ensures brand and product consistency with localized product labels (vs product images generated via generic GenAI models that haven't been fine-tuned for specific brands/products).

## Rationale
The implementation prioritizes modular, repeatable design over ad-hoc automation, establishing a blueprint for how GenAI can be operationalized across creative production workflows, balancing automation, brand safety, and human oversight. It illustrates how structured orchestration and modular AI components can standardize quality and accelerate delivery while preserving creative control.

## Architecture

    main.py                   – FastAPI web server and API endpoints  
    campaign_manager.py       – Orchestration and reporting logic  
    workers/ 
      ├─ translation.py       – Culture-fitting localized message generation    
      ├─ legal_check.py       – Locale-specific prohibited term scanning
      ├─ prompt_builder.py    – Builds contextual GenAI prompts for culture fitting images 
      ├─ generation.py        – Handles image generation / outpainting 
      ├─ composition.py       – Layout, text, and logo composition  
      ├─ brand_compliance.py  – Logo detection using OpenCV feature matching  
    ui/
      ├─ index.html           – Front-end to build briefs, monitor progress and outputs

Each worker can be replaced or scaled independently. The **CampaignManager** coordinates workflow, tracks progress, and emits JSON reports consumed by the UI.

#### Orchestration Workflow
<details>
<summary>Click to expand workflow diagram</summary>

```mermaid
graph TD
A(API) --> B(Campaign Manager)
B --> C([Needs localization?])
C -- yes --> D(Translation)
C -- no --> E(Legal Check)
D --> E
E -- pass --> F([has creative?])
E -- fail --> Y[[Job failed]]
F -- no --> G(Prompt Builder)
F -- yes --> H([Needs A/R Variations?])
G --> I(GenAI Base Image)
I --> H
H -- yes --> J(GenAI Variations)
H -- no --> K([Needs logo?])
K -- yes --> L(Layout/Composition Engine)
K -- no --> N
J --> L
H -- no
M -- yes --> L
M -- no --> N(Brand check)
L --> N
N -- pass --> Z[[Job complete]]
N -- fail --> Y
```
</details>

## Running the app

### Requirements

Create a virtual python environment and activate it
`python -m venv venv` then `source venv/bin/activate`

Install dependencies:
`pip install -r requirements.txt` 

Rename `.env.sample` to `.env` with a valid OpenAI API key:
`OPENAI_API_KEY="sk-..."` 

### Run locally
`uvicorn main:app --reload` 

Open http://localhost:8000 to access the web UI

### Inputs

Submit a campaign brief in JSON format via the UI or API (`/process_brief`).  
Each brief defines:

-   Products (id, theme, audience)
-   Regions (e.g., `en-US`, `fr-FR`, `ja-JP`)
-   Aspect ratios (`1:1`, `9:16`, `16:9`)
-   Assets (optional creative image, logo/message flags)

Note: if an input image is provided, specify if it already contains a logo and/or message

### Outputs

-   Generated creatives for all ratios/regions under `/outputs/{job_id}/`
-   `/outputs/{job_id}/report.json` with progress, status, and compliance checks
-   `/outputs/{job_id}/log.txt` with system logs (background tasks do not std out to main thread)
-   Real-time job updates available through APi via `/jobs/{job_id}`

## Architectural Principles

-   **Modularity:** Each function is encapsulated as a worker, ensuring isolation and easy upgrades.
-   **Specialization:** Tasks are routed to the most appropriate worker (generation, translation, composition, compliance).
-   **Observability:** Every step logs to a per-job report with deterministic progress tracking.
-   **Extensibility:** The OpenAI APIs can be replaced with other services/models to test different results without structural refactoring.
    

## Technical Notes

-   **Image Generation:** Uses OpenAI Image Edit API for masked outpainting. While this simplifies implementation, masks are not strictly respected, occasionally altering product placement, labels and features. Firefly Services provide superior control and brand fidelity.
    
-   **Brand Detection:** Template and feature-based matching via OpenCV (SIFT + FLANN) is functional but limited. Integrating ML-based logo detection would improve reliability for scaled deployments.
    
-   **Composition:** Typography, layout, and contrast selection are handled programmatically for now; a layout model (e.g. Photoshop Layers) could further enhance visual consistency and creative control.
    

## Limitations

-   Outpainting fidelity and product preservation depend on model behavior.
-   Brand detection is fairly simple for this version.
-   No async job queue (single process for demo simplicity).
    

## Potential Enhancements

-  Replace OpenAI image editing with Firefly Services.

-  Build a more robust composition engine that leverages Photoshop server to use layers, text layouts. 
    
-  Introduce a proper async task queue for concurrent campaign runs.
    
-   Extend the compliance layer to include text-in-image OCR, layout heuristics, brand colors and ML to detect brand features.




