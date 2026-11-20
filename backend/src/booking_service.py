"""Sample-booking domain service.

Reservation timeline:
  create  -> pending (no trained bound) or committed (goods reserved by bound)
  finish  -> completion lock: real good shots are allocated FIFO by promise
             time; early promises are served first, uncovered tail is booked
             as suspended shortfall; the reservation list becomes immutable
  sign    -> locked orders can dispatch in batches; retry with the same
             batch number never deducts twice
"""
from datetime import datetime, timezone
from fastapi import HTTPException
from sqlalchemy import select
from .models import Mould, Estimate
from .execution_models import TrialExecution, CostReview
from .workshop_models import TrialDelivery
from .booking_models import YieldModel, SampleOrder, SampleDispatch
from . import yield_ml
from .services import item, serialize, record


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def latest_model(s, mould_id):
    return s.scalar(
        select(YieldModel)
        .where(YieldModel.mould_id == mould_id)
        .order_by(YieldModel.version.desc(), YieldModel.id.desc())
    )


def _execution_of(s, estimate_id):
    return s.scalar(select(TrialExecution).where(TrialExecution.estimate_id == estimate_id))


def _legacy_delivered(s, execution_id):
    """Samples already sent through the signed-review delivery flow."""
    return sum(s.scalars(
        select(TrialDelivery.good_samples)
        .join(CostReview, TrialDelivery.review_id == CostReview.id)
        .where(CostReview.execution_id == execution_id)
    ), 0) or 0


def _new_allocated(s, execution_id):
    """Reserved + already dispatched through the booking flow."""
    rows = s.execute(
        select(SampleOrder.reserved_qty, SampleOrder.dispatched_qty)
        .join(TrialExecution, TrialExecution.estimate_id == SampleOrder.estimate_id)
        .where(TrialExecution.id == execution_id)
    ).all()
    return sum(r[0] + r[1] for r in rows)


def _physical_remaining(s, execution):
    """Goods not yet handed out physically (legacy + booking dispatches).

    Reservations are prospective claims, not consumed stock, so they do not
    reduce this number — otherwise a fully reserved order could never ship."""
    legacy = _legacy_delivered(s, execution.id)
    dispatched = sum(s.scalars(
        select(SampleOrder.dispatched_qty)
        .join(TrialExecution, TrialExecution.estimate_id == SampleOrder.estimate_id)
        .where(TrialExecution.id == execution.id)
    ), 0) or 0
    return execution.good_shots - legacy - dispatched


def available_goods(s, execution):
    """Goods still claimable by the legacy delivery flow: every booking
    commitment (reserved or already shipped) reduces the shared pool."""
    return execution.good_shots - _legacy_delivered(s, execution.id) - _new_allocated(s, execution.id)


def view(s, o):
    d = serialize(o)
    mould = s.get(Mould, o.mould_id)
    estimate = s.get(Estimate, o.estimate_id)
    d['mould_code'] = mould.code if mould else None
    d['estimate_code'] = estimate.code if estimate else None
    x = _execution_of(s, o.estimate_id)
    d['execution_id'] = x.id if x else None
    d['execution_status'] = x.status if x else None
    d['remaining_to_dispatch'] = o.reserved_qty
    d['expected_good_qty'] = int(o.planned_shots * (o.rate_snapshot or 0))
    model = latest_model(s, o.mould_id)
    d['model_version'] = model.version if model else None
    d['model_lower_rate'] = model.lower_rate if model else None
    d['batches'] = [serialize(b) for b in
                    s.scalars(select(SampleDispatch).where(SampleDispatch.order_id == o.id)
                              .order_by(SampleDispatch.id))]
    return d


# ---------- training ----------

def train_models(s, p, u):
    reports, failures = yield_ml.train_all(s, p.mould_id, z=p.z)
    saved = []
    for rep in reports:
        prev = latest_model(s, rep['mould_id'])
        version = (prev.version + 1) if prev else 1
        m = YieldModel(mould_id=rep['mould_id'], version=version,
                       lower_rate=rep['lower_rate'], sample_size=rep['sample_size'],
                       mean_rate=rep['mean_rate'], std_rate=rep['std_rate'],
                       z=rep['z'], fold_rows=rep['fold_rows'],
                       metrics=rep['metrics'], trained_at=now_iso())
        s.add(m)
        s.flush()
        saved.append(serialize(m))
        record(s, u, f'训练良品率下界 模具{rep["mould_id"]} v{version}')
    # A freshly trained bound may release orders that were suspended.
    for rep in reports:
        _evaluate_pending(s, rep['mould_id'])
    return {'trained': saved, 'failed': failures}


def yield_models(s):
    out = []
    for m in s.scalars(select(YieldModel).order_by(YieldModel.mould_id,
                                                    YieldModel.version.desc())):
        d = serialize(m)
        d['mould_code'] = item(s, Mould, m.mould_id).code
        out.append(d)
    return out


# ---------- ordering ----------

def create_order(s, p, u):
    e = s.query(Estimate).filter_by(id=p.estimate_id).with_for_update().one_or_none()
    if not e:
        raise HTTPException(404, '估单不存在')
    if e.status != 'committed':
        raise HTTPException(409, '仅已承诺机时的试模估单可在完工前订样')
    x = _execution_of(s, e.id)
    if x and x.status in ('completed', 'settled'):
        raise HTTPException(409, '试模已完工，订样名单已锁定，不能新增订样单')
    # Mould lock serializes concurrent bookings that draw on the same pool.
    mould = s.query(Mould).filter_by(id=e.mould_id).with_for_update().one()
    model = latest_model(s, mould.id)
    order = SampleOrder(order_no=p.order_no, customer=p.customer, estimate_id=e.id,
                        mould_id=mould.id, planned_shots=p.planned_shots,
                        requested_qty=p.requested_qty, note=p.note)
    if not model:
        order.status = 'pending'
        s.add(order)
        s.flush()
        record(s, u, '挂起订样单(无良品率模型) ' + order.order_no)
        return view(s, order)
    expected = int(p.planned_shots * model.lower_rate)
    promised = min(p.requested_qty, expected)
    if promised <= 0:
        order.status = 'pending'
        order.rate_snapshot = model.lower_rate
        s.add(order)
        s.flush()
        record(s, u, '挂起订样单(下界可承诺为0) ' + order.order_no)
        return view(s, order)
    # Sibling orders draw on the same future good-shot pool of this trial.
    used = sum(s.scalars(
        select(SampleOrder.reserved_qty).where(SampleOrder.estimate_id == e.id)
    ), 0) or 0
    if used + promised > expected:
        raise HTTPException(409, f'并发订样超占：本试模按下界仅可承诺 {expected} 件，已占 {used} 件')
    order.promised_qty = promised
    order.committed_qty = promised
    order.reserved_qty = promised
    order.rate_snapshot = model.lower_rate
    order.status = 'committed'
    s.add(order)
    s.flush()
    record(s, u, f'承诺订样 {order.order_no} {promised}件')
    return view(s, order)


def _evaluate_pending(s, mould_id):
    """Re-run suspended orders once a new bound exists for their mould."""
    pending = s.scalars(
        select(SampleOrder).where(SampleOrder.mould_id == mould_id,
                                  SampleOrder.status == 'pending')
        .with_for_update()
    ).all()
    model = latest_model(s, mould_id)
    if not model:
        return
    for order in pending:
        # Once the trial is finished the list is locked; late training must
        # not resurrect orders that missed the completion cut.
        x = _execution_of(s, order.estimate_id)
        if x and x.status != 'running':
            continue
        expected = int(order.planned_shots * model.lower_rate)
        promised = min(order.requested_qty, expected)
        if promised <= 0:
            continue
        used = sum(s.scalars(
            select(SampleOrder.reserved_qty).where(
                SampleOrder.estimate_id == order.estimate_id,
                SampleOrder.id != order.id)
        ), 0) or 0
        if used + promised > expected:
            continue
        order.rate_snapshot = model.lower_rate
        order.promised_qty = promised
        order.committed_qty = promised
        order.reserved_qty = promised
        order.status = 'committed'


def evaluate_pending(s, u):
    mould_ids = s.scalars(select(SampleOrder.mould_id)
                          .where(SampleOrder.status == 'pending').distinct()).all()
    for mould_id in mould_ids:
        _evaluate_pending(s, mould_id)
    record(s, u, '重评挂起订样单')


# ---------- completion lock & FIFO allocation ----------

def lock_for_execution(s, execution, u):
    """Freeze the booking list against actual good shots.

    Orders keep their promised quantity; goods are allocated earliest-promise
    first. Anything the pool cannot cover becomes suspended shortfall.
    """
    orders = s.scalars(
        select(SampleOrder)
        .where(SampleOrder.estimate_id == execution.estimate_id,
               SampleOrder.status.in_(['committed', 'locked', 'locked_short']))
        .order_by(SampleOrder.id)
        .with_for_update()
    ).all()
    if not orders or all(o.locked_at for o in orders):
        return
    pool = max(0, execution.good_shots - _legacy_delivered(s, execution.id))
    remaining = pool
    for o in orders:
        allocated = min(o.reserved_qty, remaining)
        o.shortfall_qty = o.promised_qty - allocated
        o.reserved_qty = allocated
        o.committed_qty = o.promised_qty
        o.status = 'locked_short' if o.shortfall_qty else 'locked'
        o.locked_at = now_iso()
        remaining -= allocated
    record(s, u, f'完工锁定订样名单 实绩{execution.id} 可分{pool}件')


# ---------- dispatch ----------

def dispatch(s, oid, p, u):
    o = s.query(SampleOrder).filter_by(id=oid).one_or_none()
    if not o:
        raise HTTPException(404, '订样单不存在')
    if o.status not in ('locked', 'locked_short', 'completed'):
        raise HTTPException(409, '订样单尚未完工锁定，不能发样')
    # Idempotent retry: same natural key returns the original batch and does
    # not deduct the reservation a second time.
    existing = s.scalar(select(SampleDispatch).where(
        SampleDispatch.order_id == o.id, SampleDispatch.batch_no == p.batch_no))
    if existing:
        return view(s, o), serialize(existing), True
    x = _execution_of(s, o.estimate_id)
    if not x:
        raise HTTPException(404, '关联实绩不存在')
    # Same lock order as finish(): execution first, then the order, so the
    # two transactions can never deadlock on the same good-shot pool.
    x = s.query(TrialExecution).filter_by(id=x.id).with_for_update().one()
    o = s.query(SampleOrder).filter_by(id=oid).with_for_update().one()
    if x.status != 'settled':
        raise HTTPException(409, '结算签结后才能按锁定件数分批发样')
    if p.qty > o.reserved_qty:
        raise HTTPException(409, f'本批 {p.qty} 件超过待发预留 {o.reserved_qty} 件，发样未扣件')
    # Hard invariant across every order of the same execution: only goods not
    # yet handed out physically may be shipped.
    free = _physical_remaining(s, x)
    if p.qty > free:
        raise HTTPException(409, f'良品池余量仅 {free} 件，不能超占发样')
    snap = {'order_no': o.order_no, 'customer': o.customer,
            'mould_id': o.mould_id, 'estimate_id': o.estimate_id,
            'execution_id': x.id, 'good_shots': x.good_shots,
            'legacy_delivered': _legacy_delivered(s, x.id),
            'rate_snapshot': o.rate_snapshot,
            'dispatched_at': now_iso()}
    b = SampleDispatch(order_id=o.id, batch_no=p.batch_no, qty=p.qty,
                       state='dispatched', snapshot=snap)
    s.add(b)
    o.reserved_qty -= p.qty
    o.dispatched_qty += p.qty
    if o.reserved_qty == 0:
        o.status = 'completed'
    s.flush()
    record(s, u, f'订样发样 {o.order_no} 批次{p.batch_no} {p.qty}件')
    return view(s, o), serialize(b), False


def receive(s, bid, p, u):
    b = s.query(SampleDispatch).filter_by(id=bid).with_for_update().one_or_none()
    if not b:
        raise HTTPException(404, '发样批次不存在')
    if b.state != 'dispatched':
        raise HTTPException(409, '该批次已签收，不能重复签收')
    b.state = 'received'
    b.tracking = p.tracking
    b.received_by = p.received_by
    record(s, u, f'订样批次签收 {b.batch_no}')
    return serialize(b)


# ---------- shortages ----------

def shortages(s):
    rows = s.scalars(select(SampleOrder).order_by(SampleOrder.id)).all()
    gap, suspended = [], []
    for o in rows:
        d = view(s, o)
        if o.shortfall_qty > 0:
            gap.append(d)
        if o.status == 'pending':
            suspended.append(d)
    return {'shortfall_orders': gap, 'suspended_orders': suspended}
