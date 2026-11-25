"""Single brokered DeepSeek client used by ALL three API methods.

call_api(method_adapter, stage, prompt, max_tokens) is the ONLY paid path:
  1. model gate (deepseek-flash verified via GET /models, thinking disabled);
  2. HTTP cache lookup keyed by (model, stage, sha256(request_body)) — a paid
     response is reused verbatim (free) whenever the identical request recurs;
  3. budget broker reservation BEFORE any send (conservative, PEAK prices);
  4. OpenAI SDK send with max_retries=0; ONE app-level retry on transient
     errors (timeout / 5xx / connection) with a fresh reservation; the failed
     attempt of a timeout/5xx goes to the `unknown` account (server may bill);
     auth/balance/config errors raise MethodBlocked and stop the method;
  5. settle with the actual usage fields and the response-time price bucket;
     missing usage keeps the conservative reserve in `unknown`;
  6. evidence JSONL append (request sha256, params, response id/model/usage,
     raw content, cost, cache flag) — never the API key.
"""
import fcntl
import hashlib
import json
import math
import threading
import time
from pathlib import Path

from experiments.bird.api.budget_broker import BudgetBroker, BudgetExceeded
from experiments.bird.api.config import (
    API_EVIDENCE_DIR, HTTP_CACHE_DIR, MAX_APP_RETRIES, MODEL_SELECTION_PATH,
    REQUESTED_MODEL, REQUEST_TIMEOUT_S, THINKING_CONTROL,
    THINKING_CONTROL_NOTE, load_credentials, price_bucket_now,
)

_CLIENT = None
_CLIENT_LOCK = threading.Lock()
_BROKER = None
_MODEL_STATE = {'model': None, 'checked': False}
_EVIDENCE_LOCKS = {}


class MethodBlocked(Exception):
    """Non-retryable condition that must stop the whole method.

    reason: 'blocked_model_unavailable' | 'blocked_auth' |
            'blocked_balance' | 'blocked_config'
    """

    def __init__(self, reason, detail=''):
        super().__init__(f'{reason}: {detail}'[:400])
        self.reason = reason
        self.detail = str(detail)[:300]


class ApiTransientError(Exception):
    """API failure that survived the single app-level retry."""


# --------------------------------------------------------------- model gate
def list_models():
    """FREE GET {base}/models call. Returns the raw parsed JSON."""
    client = _get_client()
    return client.models.list().model_dump()


def select_model(force=False):
    """Verify deepseek-flash (or any id containing 'flash') is offered.

    Returns the exact model id to use. Raises MethodBlocked(
    'blocked_model_unavailable') otherwise. The decision (requested vs
    listed ids) is persisted to api_evidence/model_selection.json.
    """
    if _MODEL_STATE['checked'] and not force:
        if _MODEL_STATE['model'] is None:
            raise MethodBlocked('blocked_model_unavailable',
                                'no flash model listed on /models')
        return _MODEL_STATE['model']
    raw = list_models()
    ids = sorted(m.get('id') or '' for m in raw.get('data', []))
    chosen = None
    if REQUESTED_MODEL in ids:
        chosen = REQUESTED_MODEL
    else:
        flash_like = [i for i in ids if 'flash' in i.lower()]
        if len(flash_like) == 1:
            chosen = flash_like[0]
    record = {'requested': REQUESTED_MODEL, 'listed_ids': ids,
              'selected': chosen,
              'thinking_control': THINKING_CONTROL,
              'thinking_control_note': THINKING_CONTROL_NOTE,
              'checked_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                              time.gmtime())}
    if chosen is None:
        record['decision'] = ('no flash model listed; all API methods are '
                              'blocked_model_unavailable; NO substitution')
    MODEL_SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_SELECTION_PATH.write_text(json.dumps(record, ensure_ascii=False,
                                               indent=1) + '\n',
                                    encoding='utf-8')
    _MODEL_STATE['checked'] = True
    _MODEL_STATE['model'] = chosen
    if chosen is None:
        raise MethodBlocked('blocked_model_unavailable',
                            f'listed ids: {ids}')
    return chosen


def get_model():
    if not _MODEL_STATE['checked']:
        return select_model()
    if _MODEL_STATE['model'] is None:
        raise MethodBlocked('blocked_model_unavailable',
                            'no flash model listed on /models')
    return _MODEL_STATE['model']


def _get_client():
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            from openai import OpenAI
            api_key, base_url = load_credentials()
            if not api_key:
                raise MethodBlocked('blocked_config',
                                    'no DEEPSEEK_API_KEY in env or '
                                    'project config.yaml')
            _CLIENT = OpenAI(api_key=api_key, base_url=base_url,
                             timeout=REQUEST_TIMEOUT_S, max_retries=0)
        return _CLIENT


def get_broker():
    global _BROKER
    if _BROKER is None:
        _BROKER = BudgetBroker()
    return _BROKER


# ------------------------------------------------------------ error classes
def _is_fatal(exc):
    """Auth / balance / config errors: not retryable, method must stop."""
    try:
        import openai as _o
        fatal_types = (_o.AuthenticationError, _o.PermissionDeniedError,
                       _o.BadRequestError, _o.NotFoundError,
                       _o.UnprocessableEntityError)
        if isinstance(exc, fatal_types):
            return True
    except ImportError:
        pass
    status = getattr(exc, 'status_code', None)
    if status is None:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
    return status in (400, 401, 402, 403, 404, 422)


def _fatal_reason(exc):
    status = getattr(exc, 'status_code', None)
    if status == 402:
        return 'blocked_balance'
    if status in (400, 422):
        return 'blocked_config'
    return 'blocked_auth'


def _is_transient(exc):
    """Timeout / 5xx / connection problems: one app-level retry allowed."""
    try:
        import openai as _o
        if isinstance(exc, _o.APIConnectionError):  # incl. APITimeoutError
            return True
        if isinstance(exc, _o.APIStatusError):
            return getattr(exc, 'status_code', 0) >= 500 or \
                getattr(exc, 'status_code', 0) == 429
    except ImportError:
        pass
    status = getattr(exc, 'status_code', None)
    if status is None:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
    if status is not None:
        return status >= 500 or status == 429
    return isinstance(exc, (TimeoutError, ConnectionError))


def _never_sent(exc):
    """Connection-level failure where the request did not reach the server:
    the reservation may be released (nothing can be billed)."""
    try:
        import openai as _o
        if isinstance(exc, _o.APITimeoutError):
            return False  # request sent, response not received in time
        if isinstance(exc, _o.APIConnectionError):
            return True
    except ImportError:
        pass
    return isinstance(exc, ConnectionError) and not isinstance(exc, TimeoutError)


# ---------------------------------------------------------------- estimates
def conservative_input_tokens(prompt):
    """chars/3 tokens, floored at 1.2x any tighter estimate: for English
    schema text the DeepSeek tokenizer averages ~3.5-4 chars/token, so
    chars/3 is already >= 1.2x the expected actual count."""
    return max(1, math.ceil(len(prompt) / 3.0))


# ------------------------------------------------------------------ caching
def request_cache_key(model, stage, params, prompt):
    body = {'model': model, 'stage': stage, 'params': params,
            'messages': [{'role': 'user', 'content': prompt}]}
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def cache_path(key):
    return HTTP_CACHE_DIR / f'{key}.json'


# ----------------------------------------------------------------- evidence
def _evidence_path(method, unit):
    safe = (unit or 'global').replace('/', '__')
    return API_EVIDENCE_DIR / method / f'{method}__{safe}.jsonl'


def append_evidence(method, unit, record):
    path = _evidence_path(method, unit)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _EVIDENCE_LOCKS.setdefault(str(path), threading.Lock())
    with lock:
        with path.open('a', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def redact(text):
    """Defensive redaction: never let any credential-looking env/config value
    leak into evidence. The key is never intentionally placed in evidence;
    this guards against accidental inclusion via response/error payloads."""
    api_key, _ = load_credentials()
    if api_key and api_key in text:
        text = text.replace(api_key, '[REDACTED_API_KEY]')
    return text


# ------------------------------------------------------------------ result
class CallResult:
    __slots__ = ('content', 'response_id', 'response_model', 'usage',
                 'cached', 'req_id', 'cost_cny', 'bucket', 'attempts',
                 'reserve_cny', 'cache_key')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def call_api(method_adapter, stage, prompt, max_tokens, temperature=0.1):
    """Brokered paid call. method_adapter must expose: api_method (name),
    api_unit (unit label), api_qid (question id, may be empty).

    Returns CallResult. Raises MethodBlocked / ApiTransientError /
    BudgetExceeded.
    """
    method = getattr(method_adapter, 'api_method', 'unknown')
    unit = getattr(method_adapter, 'api_unit', '')
    qid = getattr(method_adapter, 'api_qid', '')
    model = get_model()
    params = {'temperature': temperature, 'max_tokens': int(max_tokens),
              'extra_body': dict(THINKING_CONTROL)}
    key = request_cache_key(model, stage, params, prompt)
    cpath = cache_path(key)
    broker = get_broker()

    ev_base = {'ts_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'method': method, 'stage': stage, 'unit': unit, 'qid': qid,
               'request': {'model': model, **{k: v for k, v in params.items()
                                              if k != 'extra_body'},
                           'thinking': THINKING_CONTROL,
                           'timeout_s': REQUEST_TIMEOUT_S,
                           'sdk_max_retries': 0,
                           'app_max_retries': MAX_APP_RETRIES},
               'prompt_sha256': hashlib.sha256(
                   prompt.encode('utf-8')).hexdigest(),
               'prompt_chars': len(prompt),
               'est_input_tokens': conservative_input_tokens(prompt),
               'cache_key': key}

    # ---- cache hit: the paid response is reused, zero new cost ----
    if cpath.is_file():
        try:
            cached = json.loads(cpath.read_text(encoding='utf-8'))
            rec = dict(ev_base, cached=True, req_id=cached.get('req_id', ''),
                       response=cached.get('response'),
                       content=cached.get('content'),
                       cost_cny=0.0, bucket=cached.get('bucket'),
                       attempts=0, error=None)
            append_evidence(method, unit, rec)
            return CallResult(content=cached.get('content'),
                              response_id=(cached.get('response') or {})
                              .get('id'),
                              response_model=(cached.get('response') or {})
                              .get('model'),
                              usage=(cached.get('response') or {}).get('usage'),
                              cached=True,
                              req_id=cached.get('req_id', ''),
                              cost_cny=0.0,
                              bucket=cached.get('bucket'),
                              attempts=0,
                              reserve_cny=0.0,
                              cache_key=key)
        except (json.JSONDecodeError, OSError):
            pass  # corrupt cache entry: fall through to a fresh paid call

    # ---- paid path: reserve -> send -> settle ----
    client = _get_client()
    est_in = conservative_input_tokens(prompt)
    last_exc = None
    for attempt in range(1, MAX_APP_RETRIES + 2):
        res = broker.reserve(method, stage, est_in, int(max_tokens),
                             unit=unit)
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=params['temperature'],
                max_tokens=params['max_tokens'],
                extra_body=dict(THINKING_CONTROL))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_fatal(exc):
                broker.release(res['req_id'], f'fatal:{type(exc).__name__}')
                reason = _fatal_reason(exc)
                ev = dict(ev_base, cached=False, req_id=res['req_id'],
                          attempts=attempt, cost_cny=0.0, bucket=None,
                          error=f'{type(exc).__name__}: {str(exc)[:200]}')
                append_evidence(method, unit, ev)
                raise MethodBlocked(reason, f'{type(exc).__name__}: '
                                    f'{str(exc)[:200]}') from exc
            # transient: timeout/5xx may still be billed -> unknown;
            # a never-sent connection error releases the reservation.
            if _never_sent(exc):
                broker.release(res['req_id'],
                               f'never_sent:{type(exc).__name__}')
            else:
                broker.mark_unknown(res['req_id'],
                                    f'transient:{type(exc).__name__}')
            if attempt <= MAX_APP_RETRIES:
                time.sleep(2)
                continue  # fresh reservation for the retry
            ev = dict(ev_base, cached=False, req_id=res['req_id'],
                      attempts=attempt, cost_cny=0.0, bucket=None,
                      error=f'{type(exc).__name__}: {str(exc)[:200]}')
            append_evidence(method, unit, ev)
            raise ApiTransientError(
                f'{stage}: API failed after {attempt} attempts: '
                f'{type(exc).__name__}') from exc

        # success: settle with the ACTUAL usage at the response-time bucket
        content = resp.choices[0].message.content
        usage = None
        if getattr(resp, 'usage', None) is not None:
            try:
                usage = resp.usage.model_dump()
            except Exception:  # noqa: BLE001
                usage = {'prompt_tokens': getattr(resp.usage,
                                                  'prompt_tokens', None),
                         'completion_tokens': getattr(resp.usage,
                                                      'completion_tokens',
                                                      None),
                         'total_tokens': getattr(resp.usage, 'total_tokens',
                                                 None)}
        bucket = price_bucket_now()
        pt = int((usage or {}).get('prompt_tokens') or 0)
        ct = int((usage or {}).get('completion_tokens') or 0)
        if usage and (usage.get('prompt_tokens') is None or
                      usage.get('completion_tokens') is None):
            usage = None  # incomplete usage fields -> unknown account
        settled = broker.settle(res['req_id'], usage=usage, bucket=bucket,
                                response_id=getattr(resp, 'id', None),
                                response_model=getattr(resp, 'model', None))
        response_rec = {'id': getattr(resp, 'id', None),
                        'model': getattr(resp, 'model', None),
                        'usage': usage,
                        'finish_reason': getattr(resp.choices[0],
                                                 'finish_reason', None)}
        cost = (settled or {}).get('cost_cny')
        # durable HTTP cache: an identical future request reuses this paid
        # response for free; a parser fix never invalidates it.
        try:
            HTTP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cpath.with_suffix('.tmp')
            tmp.write_text(json.dumps(
                {'req_id': res['req_id'], 'request': ev_base['request'],
                 'prompt_sha256': ev_base['prompt_sha256'],
                 'response': response_rec, 'content': content,
                 'bucket': bucket, 'settled': settled},
                ensure_ascii=False), encoding='utf-8')
            tmp.replace(cpath)
        except OSError:
            pass
        ev = dict(ev_base, cached=False, req_id=res['req_id'],
                  response=response_rec, content=content,
                  cost_cny=cost, bucket=bucket, attempts=attempt,
                  http_seconds=round(time.time() - t0, 3), error=None)
        append_evidence(method, unit, ev)
        return CallResult(content=content,
                          response_id=response_rec['id'],
                          response_model=response_rec['model'],
                          usage=usage, cached=False, req_id=res['req_id'],
                          cost_cny=cost, bucket=bucket, attempts=attempt,
                          reserve_cny=res['reserve_cny'], cache_key=key)
    raise ApiTransientError(f'{stage}: exhausted attempts ({last_exc})')
