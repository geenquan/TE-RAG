"""Cross-process-safe global budget broker for ALL paid DeepSeek calls.

Hard invariant (enforced atomically BEFORE any send):
    spent + unknown + reserved + new_reserve <= cap (CNY 20.0)

Design:
- Ledger JSON at api_evidence/budget_ledger.json; every mutation happens under
  an exclusive fcntl flock on api_evidence/budget_ledger.lock, with the file
  replaced atomically (tmp + os.replace) so a crash never leaves a torn file.
- A request moves through states: reserved -> settled | unknown | released.
  * reserved: conservative hold  = est_input_tokens @ PEAK input price
                                 + max_tokens    @ PEAK output price.
  * settled:  actual cost from the response usage fields at the price bucket
              of the response time (off-peak discount allowed at settle).
  * unknown:  the response carried no usage (or a timeout/5xx means the server
              may still bill it): the FULL conservative reserve stays in the
              unknown account forever; it is never freed.
  * released: failure before send (connection refused / fatal 4xx): no charge.
- Every mutation appends one line to api_evidence/budget_audit.jsonl (JSON
  audit trail) and refreshes api_evidence/budget_state.json (streaming state).
- The API key never appears anywhere in this module's outputs.
"""
import fcntl
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from experiments.bird.api.config import (
    API_EVIDENCE_DIR, AUDIT_PATH, BUDGET_CAP_CNY, BUDGET_STATE_PATH,
    LEDGER_LOCK_PATH, LEDGER_PATH, PRICE_CNY_PER_M,
)

_EPS = 1e-9


class BudgetExceeded(Exception):
    """The new reservation would break the global cap invariant."""


class DuplicateRequest(Exception):
    """req_id already tracked (reserved/settled/unknown)."""


def reserve_cost_cny(est_input_tokens, max_tokens):
    """Conservative reservation at PEAK prices (CNY)."""
    p = PRICE_CNY_PER_M['peak']
    return (est_input_tokens * p['input'] + max_tokens * p['output']) / 1e6


def actual_cost_cny(prompt_tokens, completion_tokens, bucket):
    p = PRICE_CNY_PER_M[bucket]
    return (prompt_tokens * p['input'] + completion_tokens * p['output']) / 1e6


class BudgetBroker:
    def __init__(self, cap_cny=BUDGET_CAP_CNY, evidence_dir=None):
        self.cap = float(cap_cny)
        self.dir = Path(evidence_dir) if evidence_dir else API_EVIDENCE_DIR
        self.ledger_path = self.dir / LEDGER_PATH.name
        self.lock_path = self.dir / LEDGER_LOCK_PATH.name
        self.audit_path = self.dir / AUDIT_PATH.name
        self.state_path = self.dir / BUDGET_STATE_PATH.name
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ io
    def _load(self):
        if not self.ledger_path.is_file():
            return {'cap_cny': self.cap, 'requests': {}}
        try:
            led = json.loads(self.ledger_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            # A previous atomic write never leaves torn files; an unreadable
            # ledger is a hard stop, not a reset (never free money by accident).
            raise RuntimeError(f'budget ledger unreadable: {self.ledger_path}')
        led.setdefault('requests', {})
        return led

    def _write(self, led):
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), prefix='.ledger_')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(led, f, ensure_ascii=False, indent=1, default=str)
                f.write('\n')
            os.replace(tmp, self.ledger_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _audit(self, event, payload):
        line = {'ts_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'event': event, **payload}
        with self.audit_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(line, ensure_ascii=False, default=str) + '\n')

    # -------------------------------------------------------------- totals
    @staticmethod
    def _totals(led):
        spent = unknown = reserved = 0.0
        by_method = {}
        for r in led['requests'].values():
            m = r.get('method', '?')
            cost = float(r.get('settled', {}).get('cost_cny', 0) or 0) \
                if r.get('status') == 'settled' else 0.0
            if r.get('status') == 'settled':
                spent += cost
                slot = 'spent_cny'
            elif r.get('status') == 'unknown':
                unknown += float(r.get('reserve_cny', 0))
                slot = 'unknown_cny'
            elif r.get('status') == 'reserved':
                reserved += float(r.get('reserve_cny', 0))
                slot = 'reserved_cny'
            else:
                continue
            d = by_method.setdefault(m, {'spent_cny': 0.0, 'unknown_cny': 0.0,
                                         'reserved_cny': 0.0,
                                         'n_calls': 0, 'n_settled': 0,
                                         'n_unknown': 0})
            d[slot] += cost if slot == 'spent_cny' else float(r.get('reserve_cny', 0))
            d['n_calls'] += 1
            if r.get('status') == 'settled':
                d['n_settled'] += 1
            if r.get('status') == 'unknown':
                d['n_unknown'] += 1
        return {'spent_cny': round(spent, 6), 'unknown_cny': round(unknown, 6),
                'reserved_cny': round(reserved, 6), 'by_method': by_method}

    def state(self):
        led = self._load()
        t = self._totals(led)
        committed = t['spent_cny'] + t['unknown_cny'] + t['reserved_cny']
        return {'cap_cny': self.cap,
                'spent_cny': t['spent_cny'],
                'unknown_cny': t['unknown_cny'],
                'reserved_cny': t['reserved_cny'],
                'committed_cny': round(committed, 6),
                'remaining_cny': round(self.cap - committed, 6),
                'n_requests': len(led['requests']),
                'by_method': t['by_method']}

    def _stream_state(self):
        st = self.state()
        st['updated_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                             time.gmtime())
        self._write_any(self.state_path, st)
        return st

    def _write_any(self, path, payload):
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), prefix='.state_')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=1,
                          default=str)
                f.write('\n')
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # ------------------------------------------------------------- locking
    class _Locked:
        def __init__(self, broker):
            self.broker = broker

        def __enter__(self):
            self.broker.dir.mkdir(parents=True, exist_ok=True)
            self.fh = open(self.broker.lock_path, 'a+', encoding='utf-8')
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
            self.led = self.broker._load()
            return self.led

        def __exit__(self, *exc):
            try:
                if exc[0] is None:
                    self.led['cap_cny'] = self.broker.cap
                    self.broker._write(self.led)
            finally:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
                self.fh.close()
            return False

    # -------------------------------------------------------------- api
    def reserve(self, method, stage, est_input_tokens, max_tokens, unit='',
                req_id=None):
        """Atomically check the cap invariant and hold a conservative reserve.

        est_input_tokens must already be conservative (chars/3, at least
        1.2x the best estimate). Raises BudgetExceeded when the invariant
        would break; the caller MUST NOT send anything in that case.
        """
        req_id = req_id or f'{stage}-{uuid.uuid4().hex[:16]}'
        cost = reserve_cost_cny(est_input_tokens, max_tokens)
        with self._Locked(self) as led:
            prev = led['requests'].get(req_id)
            if prev and prev.get('status') in ('reserved', 'settled',
                                               'unknown'):
                raise DuplicateRequest(req_id)
            t = self._totals(led)
            committed = t['spent_cny'] + t['unknown_cny'] + t['reserved_cny']
            if committed + cost > self.cap + _EPS:
                snap = {'committed_cny': round(committed, 6),
                        'requested_reserve_cny': round(cost, 6),
                        'cap_cny': self.cap, **{k: t[k] for k in
                                                ('spent_cny', 'unknown_cny',
                                                 'reserved_cny')}}
                self._audit('reserve_refused', {'req_id': req_id,
                                                'method': method,
                                                'stage': stage, **snap})
                self._stream_state()
                raise BudgetExceeded(
                    f'cap invariant: {snap["committed_cny"]} committed + '
                    f'{snap["requested_reserve_cny"]} new > {self.cap} CNY')
            led['requests'][req_id] = {
                'req_id': req_id, 'method': method, 'stage': stage,
                'unit': unit, 'status': 'reserved',
                'est_input_tokens': int(est_input_tokens),
                'max_tokens': int(max_tokens),
                'reserve_cny': round(cost, 6),
                'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                time.gmtime())}
        self._audit('reserve', {'req_id': req_id, 'method': method,
                                'stage': stage, 'unit': unit,
                                'est_input_tokens': int(est_input_tokens),
                                'max_tokens': int(max_tokens),
                                'reserve_cny': round(cost, 6)})
        self._stream_state()
        return {'req_id': req_id, 'reserve_cny': round(cost, 6)}

    def settle(self, req_id, usage=None, bucket='peak', response_id=None,
               response_model=None, billed_unknown_reason=None):
        """Settle a reservation with the ACTUAL usage returned by the API.

        usage: {'prompt_tokens': int, 'completion_tokens': int} (other fields
        preserved verbatim). Missing usage -> the full conservative reserve
        stays in the `unknown` account (never freed).
        """
        with self._Locked(self) as led:
            r = led['requests'].get(req_id)
            if r is None:
                raise KeyError(f'settle: unknown req_id {req_id}')
            if r['status'] == 'settled':
                return r['settled']
            if r['status'] == 'released':
                raise RuntimeError(f'settle after release: {req_id}')
            if usage is None:
                r['status'] = 'unknown'
                r['unknown_reason'] = (billed_unknown_reason or
                                       'usage_missing')
                settled = None
            else:
                pt = int(usage.get('prompt_tokens') or 0)
                ct = int(usage.get('completion_tokens') or 0)
                cost = actual_cost_cny(pt, ct, bucket)
                r['status'] = 'settled'
                settled = {'usage': usage, 'bucket': bucket,
                           'cost_cny': round(cost, 6),
                           'response_id': response_id,
                           'response_model': response_model,
                           'settled_at_utc': time.strftime(
                               '%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                r['settled'] = settled
        self._audit('settle', {'req_id': req_id,
                               'method': r.get('method'),
                               'stage': r.get('stage'),
                               'status': r['status'],
                               'cost_cny': (settled or {}).get('cost_cny'),
                               'bucket': bucket, 'usage': usage})
        self._stream_state()
        return settled

    def release(self, req_id, reason):
        """Release a reservation without charge (failure BEFORE any send, or
        a fatal 4xx rejection). Timed-out / 5xx attempts are NEVER released:
        the server may still bill them, so they go to `unknown` instead."""
        with self._Locked(self) as led:
            r = led['requests'].get(req_id)
            if r is None:
                raise KeyError(f'release: unknown req_id {req_id}')
            if r['status'] in ('reserved',):
                r['status'] = 'released'
                r['release_reason'] = str(reason)[:200]
        self._audit('release', {'req_id': req_id,
                                'method': r.get('method'),
                                'stage': r.get('stage'),
                                'reason': str(reason)[:200]})
        self._stream_state()

    def mark_unknown(self, req_id, reason):
        """Timed-out or 5xx attempt: keep the conservative reserve in the
        unknown account (the request may still be billed server-side)."""
        return self.settle(req_id, usage=None, bucket='peak',
                           billed_unknown_reason=str(reason)[:200])
