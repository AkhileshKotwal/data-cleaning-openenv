"""
server/app.py — FastAPI server for Data Cleaning OpenEnv v4.0
"""

import os
import sys
import uuid
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Any, Dict

from models import DataCleaningAction, DataCleaningObservation, DataCleaningState
from server.environment import DataCleaningEnvironment, _clamp_reward

app = FastAPI(
    title="Data Cleaning OpenEnv",
    description=(
        "An RL environment where agents learn to clean messy datasets. "
        "Features 4 tasks, adversarial corruption mode, per-session isolation, "
        "and a catastrophic penalty system."
    ),
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[str, DataCleaningEnvironment] = {}
VALID_TASKS = ["fix_nulls", "fix_types", "full_clean", "deceptive_clean"]


def _get_session(session_id: str) -> DataCleaningEnvironment:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call POST /reset first.")
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Required OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "active_sessions": len(_sessions), "version": "4.0.0"}


@app.get("/metadata")
def metadata():
    """Required by openenv-core validator: GET /metadata must return name and description."""
    return {
        "name": "data-cleaning-env",
        "description": (
            "An RL environment where agents learn to clean messy real-world datasets. "
            "Features 4 tasks (easy to expert), adversarial corruption mode, "
            "catastrophic penalties for destroying critical data, and dense reward signals."
        ),
        "version": "4.0.0",
        "author": "Akhilesh Kotwal",
        "tasks": VALID_TASKS,
        "modes": ["standard", "adversarial"],
    }


@app.get("/schema")
def schema():
    """Required by openenv-core validator: GET /schema must return action, observation, state schemas."""
    return {
        "action": DataCleaningAction.model_json_schema(),
        "observation": DataCleaningObservation.model_json_schema(),
        "state": DataCleaningState.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(request: Request):
    """Required by openenv-core validator: POST /mcp must return JSON-RPC {jsonrpc: '2.0'}."""
    try:
        body = await request.body()
        req_data = json.loads(body) if body else {}
    except Exception:
        req_data = {}

    method = req_data.get("method", "")
    req_id = req_data.get("id", 1)

    if method == "tools/list":
        tools = [
            {"name": "reset",  "description": "Reset the environment for a task."},
            {"name": "step",   "description": "Take one cleaning action."},
            {"name": "state",  "description": "Get current environment state."},
            {"name": "health", "description": "Check server health."},
        ]
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}

    # Default: return a valid JSON-RPC 2.0 response
    return {"jsonrpc": "2.0", "id": req_id, "result": {}}


# ---------------------------------------------------------------------------
# Core environment endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset(raw: Request, task_id: str = "fix_nulls"):
    try:
        try:
            body = await raw.json()
            if not isinstance(body, dict):
                body = {}
        except Exception:
            body = {}
        tid = body.get("task_id", task_id)
        if tid not in VALID_TASKS:
            raise HTTPException(status_code=400, detail=f"Unknown task. Choose from {VALID_TASKS}")
        session_id = str(uuid.uuid4())
        env = DataCleaningEnvironment(
            task_id=tid,
            adversarial=body.get("adversarial", False),
            adversary_difficulty=body.get("adversary_difficulty", 0.3),
            seed=body.get("seed", 42),
        )
        obs = env.reset()
        _sessions[session_id] = env
        result = obs.model_dump()
        result["session_id"] = session_id
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class StepRequest(BaseModel):
    session_id: str
    action_type: str
    column: Optional[str] = None
    strategy: Optional[str] = None
    dtype: Optional[str] = None
    method: Optional[str] = None
    metadata: Dict[str, Any] = {}
    task_id: str = "fix_nulls"


@app.post("/step")
def step(request: StepRequest):
    try:
        env = _get_session(request.session_id)
        action = DataCleaningAction(
            action_type=request.action_type,
            column=request.column,
            strategy=request.strategy,
            dtype=request.dtype,
            method=request.method,
            metadata=request.metadata,
        )
        obs, reward, done, info = env.step(action)
        if done:
            _sessions.pop(request.session_id, None)
        # Hard clamp: reward must be strictly between 0 and 1 (validator requirement)
        reward_clamped = _clamp_reward(reward)
        result = obs.model_dump()
        result["reward"] = reward_clamped
        result["done"] = bool(done)
        # Clamp score inside info dict as well
        if "score" in info:
            from server.environment import _clamp as _env_clamp
            info["score"] = _env_clamp(float(info["score"]))
        result["info"] = info
        result["session_id"] = request.session_id
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state(session_id: str):
    try:
        return _get_session(session_id).state().model_dump()
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "fix_nulls",       "difficulty": "easy",   "description": "Fill all missing values.",                              "max_steps": 10},
            {"id": "fix_types",       "difficulty": "medium", "description": "Fix data types + remove duplicates.",                   "max_steps": 15},
            {"id": "full_clean",      "difficulty": "hard",   "description": "Full pipeline: nulls, types, duplicates, outliers, text.", "max_steps": 20},
            {"id": "deceptive_clean", "difficulty": "expert", "description": "Conflicting signals + catastrophic penalties.",          "max_steps": 15},
        ],
        "modes": {"standard": "Normal mode.", "adversarial": "Enemy re-corrupts your best column after every step."},
        "action_cost": 0.04,
        "version": "4.0.0",
    }


@app.get("/web", response_class=HTMLResponse)
def web_demo():
    return HTMLResponse(content=WEB_HTML)


WEB_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Cleaning Lab</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh}
.header{background:#161b22;border-bottom:1px solid #30363d;padding:14px 24px;display:flex;align-items:center;gap:12px}
.header-icon{font-size:24px}
.header-title{font-size:18px;font-weight:600;color:#e6edf3}
.header-sub{font-size:12px;color:#8b949e;margin-left:4px}
.badge{background:#21262d;border:1px solid #30363d;border-radius:20px;padding:3px 10px;font-size:11px;color:#8b949e;margin-left:8px}
.badge.running{border-color:#238636;color:#3fb950}
.layout{display:grid;grid-template-columns:280px 1fr 300px;height:calc(100vh - 53px)}
.panel{border-right:1px solid #30363d;overflow-y:auto;padding:16px}
.panel:last-child{border-right:none;border-left:1px solid #30363d}
.section{margin-bottom:20px}
.section-title{font-size:11px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px}
.task-card{border:1px solid #30363d;border-radius:8px;padding:12px;margin-bottom:8px;cursor:pointer;transition:all .15s;background:#161b22}
.task-card:hover{border-color:#58a6ff;background:#1f2937}
.task-card.active{border-color:#238636;background:#0d2818}
.task-header{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.difficulty-pill{font-size:10px;padding:2px 7px;border-radius:12px;font-weight:500}
.pill-easy{background:#0d2818;color:#3fb950;border:1px solid #238636}
.pill-medium{background:#2d1f00;color:#e3b341;border:1px solid #9e6a03}
.pill-hard{background:#2d1000;color:#f78166;border:1px solid #b22222}
.pill-expert{background:#2d0b26;color:#d2a8ff;border:1px solid #8957e5}
.task-name{font-size:14px;font-weight:500;color:#e6edf3}
.task-desc{font-size:12px;color:#8b949e;line-height:1.4}
.start-btn{width:100%;padding:10px;border-radius:8px;border:none;background:#238636;color:white;font-size:14px;font-weight:600;cursor:pointer;transition:background .15s}
.start-btn:hover{background:#2ea043}
.score-panel{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;display:flex;align-items:center;gap:16px;margin-bottom:16px}
.score-ring{width:80px;height:80px;flex-shrink:0;position:relative}
.score-ring svg{transform:rotate(-90deg)}
.score-text{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center}
.score-pct{font-size:20px;font-weight:700;color:#3fb950}
.score-word{font-size:10px;color:#8b949e}
.score-info{flex:1}
.score-title{font-size:16px;font-weight:600;color:#e6edf3;margin-bottom:4px}
.score-sub{font-size:12px;color:#8b949e;line-height:1.5}
.msg{padding:10px 14px;border-radius:8px;font-size:13px;line-height:1.5;margin-bottom:12px;border-left:3px solid}
.msg-info{background:#0d1f3a;border-color:#1f6feb;color:#79c0ff}
.msg-success{background:#0d2818;border-color:#238636;color:#3fb950}
.log-entry{font-size:11px;font-family:'Courier New',monospace;padding:5px 0;border-bottom:1px solid #21262d;color:#8b949e}
.log-start{color:#58a6ff}.log-step{color:#3fb950}.log-end{color:#e3b341}
.empty-state{text-align:center;padding:40px 20px;color:#8b949e;font-size:13px}
</style>
</head>
<body>
<div class="header">
  <span class="header-icon">🧹</span>
  <span class="header-title">Data Cleaning Lab</span>
  <span class="header-sub">AI Training Environment</span>
  <span class="badge running">v4.0</span>
</div>
<div class="layout">
  <div class="panel">
    <div class="section">
      <div class="section-title">Choose a challenge</div>
      <div class="task-card active" onclick="selectedTask='fix_nulls'">
        <div class="task-header"><span class="difficulty-pill pill-easy">Easy</span></div>
        <div class="task-name">Fill the gaps</div>
        <div class="task-desc">Some values are missing. Figure out the best way to fill them in.</div>
      </div>
      <div class="task-card" onclick="selectedTask='fix_types'">
        <div class="task-header"><span class="difficulty-pill pill-medium">Medium</span></div>
        <div class="task-name">Fix the mix-up</div>
        <div class="task-desc">Data types are wrong and rows are duplicated.</div>
      </div>
      <div class="task-card" onclick="selectedTask='full_clean'">
        <div class="task-header"><span class="difficulty-pill pill-hard">Hard</span></div>
        <div class="task-name">Deep clean</div>
        <div class="task-desc">Everything is messy — missing values, outliers, bad formatting.</div>
      </div>
      <div class="task-card" onclick="selectedTask='deceptive_clean'">
        <div class="task-header"><span class="difficulty-pill pill-expert">Expert</span></div>
        <div class="task-name">The trap</div>
        <div class="task-desc">Things are not what they seem. One wrong move ends the game.</div>
      </div>
    </div>
    <button class="start-btn" onclick="startEpisode()">▶  Start cleaning</button>
  </div>
  <div class="panel">
    <div class="score-panel">
      <div class="score-ring">
        <svg width="80" height="80" viewBox="0 0 80 80">
          <circle cx="40" cy="40" r="34" fill="none" stroke="#21262d" stroke-width="8"/>
          <circle id="score-arc" cx="40" cy="40" r="34" fill="none" stroke="#238636" stroke-width="8"
            stroke-dasharray="213.6" stroke-dashoffset="213.6" stroke-linecap="round"/>
        </svg>
        <div class="score-text">
          <div class="score-pct" id="score-pct">0%</div>
          <div class="score-word">clean</div>
        </div>
      </div>
      <div class="score-info">
        <div class="score-title">Data Cleaning Lab</div>
        <div class="score-sub" id="score-sub">Pick a challenge and press Start cleaning.</div>
      </div>
    </div>
    <div id="msg-area"></div>
  </div>
  <div class="panel">
    <div class="section-title">Episode log</div>
    <div id="log-area"><div class="empty-state">Actions will appear here</div></div>
  </div>
</div>
<script>
let selectedTask = 'fix_nulls';
let sessionId = null;
function addLog(text, cls) {
  const area = document.getElementById('log-area');
  if (area.querySelector('.empty-state')) area.innerHTML = '';
  const el = document.createElement('div');
  el.className = 'log-entry ' + cls;
  el.textContent = text;
  area.prepend(el);
}
async function startEpisode() {
  const r = await fetch('/reset', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({task_id: selectedTask, adversarial: false, seed: 42})});
  const obs = await r.json();
  sessionId = obs.session_id;
  const pct = Math.round((obs.score_so_far||0)*100);
  document.getElementById('score-pct').textContent = pct + '%';
  document.getElementById('score-sub').textContent = 'Episode started for ' + selectedTask;
  addLog('[START] task=' + selectedTask, 'log-start');
}
</script>
</body>
</html>"""


def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
        workers=int(os.getenv("WORKERS", "2")),
    )


if __name__ == "__main__":
    main()
