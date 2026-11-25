"""Shared configuration for the BIRD API methods (DenseDeepSeek, KaSLA-Adapted,
LinkAlign-SingleDB-Adapted).

Binding rules (CONTRACTS6.md "API rules (hard)"):
- Global hard budget cap: CNY 20.0 across ALL paid calls of ALL methods,
  including in-flight reservations and unknown-billed responses.
- Reservation is always at PEAK prices; settlement may use the off-peak
  discount bucket (UTC+8 00:30-08:30 off-peak, else peak) based on the
  response time.
- Credentials are read at runtime ONLY: env DEEPSEEK_API_KEY first, else the
  project-root config.yaml is parsed in memory (never printed, never copied,
  never written to any evidence file).
- Model: deepseek-flash (verified available on GET /models by the Lead).
  Thinking is explicitly disabled with the documented chat.completions
  parameter extra_body={"thinking": {"type": "disabled"}}
  (api-docs.deepseek.com guides/first_api_call + create-chat-completion).
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = Path(os.environ.get('TERAG_OUTPUT', str(PROJECT_ROOT / 'results/bird_api')))
CONFIG_YAML = PROJECT_ROOT / 'config.yaml'

API_EVIDENCE_DIR = RUN_ROOT / 'api_evidence'
HTTP_CACHE_DIR = API_EVIDENCE_DIR / 'http_cache'
DENSE_CACHE_DIR = API_EVIDENCE_DIR / 'dense_cache'
LEDGER_PATH = API_EVIDENCE_DIR / 'budget_ledger.json'
LEDGER_LOCK_PATH = API_EVIDENCE_DIR / 'budget_ledger.lock'
AUDIT_PATH = API_EVIDENCE_DIR / 'budget_audit.jsonl'
BUDGET_STATE_PATH = API_EVIDENCE_DIR / 'budget_state.json'
MODEL_SELECTION_PATH = API_EVIDENCE_DIR / 'model_selection.json'

# ---- budget (CNY per 1M tokens; contract-fixed, conservative) ----
BUDGET_CAP_CNY = 20.0
PRICE_CNY_PER_M = {
    'peak': {'input': 2.0, 'output': 8.0},
    'offpeak': {'input': 1.0, 'output': 4.0},
}
OFFPEAK_UTC8 = (0, 30)   # off-peak window starts 00:30 UTC+8 ...
OFFPEAK_END_UTC8 = (8, 30)  # ... and ends 08:30 UTC+8 (contract rule)

# ---- model wiring ----
REQUESTED_MODEL = 'deepseek-flash'
# Documented DeepSeek chat.completions thinking toggle; the official docs'
# example shows {"thinking": {"type": "enabled"}}, so "disabled" is the
# documented way to force the non-thinking mode.
THINKING_CONTROL = {'thinking': {'type': 'disabled'}}
THINKING_CONTROL_NOTE = (
    "extra_body={'thinking': {'type': 'disabled'}} passed to "
    "chat.completions.create; per api-docs.deepseek.com (guides/first_api_call "
    "example and api/create-chat-completion reference) this is the documented "
    "thinking-mode toggle; deepseek-flash defaults to thinking ON, so it is "
    "explicitly disabled for every call.")
REQUEST_TIMEOUT_S = 60.0     # same as the reference implementations
MAX_APP_RETRIES = 1          # one app-level retry on transient errors only

# MiniLM snapshot for the dense stages (same cached revision/weights the
# workspace dense_retriever resolves to; loaded with local_files_only=True
# because sentence_transformers is broken in the pinned env).
MINILM_SNAPSHOT = ('/Users/apple/.cache/huggingface/hub/models--sentence-'
                   'transformers--paraphrase-multilingual-MiniLM-L12-v2/'
                   'snapshots/'
                   'e8f8c211226b894fcb81acc59f3b34ba3efd5f42')
MINILM_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

API_CONCURRENCY = 2         # starts at 2, hard cap 4 (contract)


def load_credentials():
    """Return (api_key, base_url). Env first, then project config.yaml.

    The key is only ever returned to the OpenAI client constructor; it is
    never printed, logged, or persisted by any caller.
    """
    api_key = os.environ.get('DEEPSEEK_API_KEY', '').strip() or None
    base_url = os.environ.get('DEEPSEEK_BASE_URL', '').strip() or None
    if not api_key or not base_url:
        try:
            import yaml
            cfg = yaml.safe_load(open(CONFIG_YAML, encoding='utf-8')) or {}
            if not api_key:
                api_key = (cfg.get('DEEPSEEK_API_KEY') or '').strip() or None
            if not base_url:
                base_url = (cfg.get('DEEPSEEK_BASE_URL') or '').strip() or None
        except FileNotFoundError:
            pass
    if not base_url:
        base_url = 'https://api.deepseek.com'
    return api_key, base_url


def price_bucket_now(now=None):
    """Contract price bucket from the response time: UTC+8 00:30-08:30 is
    off-peak, everything else is peak."""
    import datetime as _dt
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    utc8 = (now + _dt.timedelta(hours=8))
    minutes = utc8.hour * 60 + utc8.minute
    start = OFFPEAK_UTC8[0] * 60 + OFFPEAK_UTC8[1]
    end = OFFPEAK_END_UTC8[0] * 60 + OFFPEAK_END_UTC8[1]
    return 'offpeak' if start <= minutes < end else 'peak'
