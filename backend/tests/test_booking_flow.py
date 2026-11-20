import os
import tempfile
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

_tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
os.environ['DATABASE_URL'] = 'sqlite:///' + _tmp.name
os.environ['REDIS_URL'] = 'redis://localhost:6379/0'
os.environ['JWT_SECRET'] = 'test-secret'
os.environ['SEED_DEMO'] = 'false'

from fastapi import HTTPException
from src.db import Base, engine, Session
from src.models import User, Mould, MachineSlot, Estimate
from src.execution_models import TrialExecution, CostReview
from src.workshop_models import TrialDelivery
from src.booking_models import YieldModel, SampleOrder, SampleDispatch
from src.booking_schemas import TrainInput, OrderInput, BatchInput, ReceiptInput
from src import booking_service as booking
from src import execution_service as exec_svc
from src.execution_schemas import StartTrial, FinishTrial, ReviewInput
from src.workshop_schemas import DeliveryInput


class Actor:
    def __init__(self, email, role):
        self.email = email
        self.role = role


ADMIN = Actor('admin@ruimo.demo', 'admin')
EST = Actor('estimator@ruimo.demo', 'estimator')


@pytest.fixture(scope='module', autouse=True)
def database():
    Base.metadata.create_all(engine)
    with Session() as s:
        for role in ('admin', 'estimator', 'viewer'):
            s.add(User(email=f'{role}@ruimo.demo', role=role, password_hash='x'))
        s.add(Mould(code='MD-A', name='训练充分模具', cavities=8, slides=4, steel='S136', weight_kg=800))
        s.add(Mould(code='MD-B', name='样本不足模具', cavities=2, slides=1, steel='P20', weight_kg=300))
        s.add(MachineSlot(machine='M280', start_hour='2026-11-18 08:00',
                          capacity_hours=200, reserved_hours=0))
        s.commit()
    yield
    Base.metadata.drop_all(engine)
    engine.dispose()


def _mould(s, code):
    return s.query(Mould).filter_by(code=code).one()


def _slot(s):
    return s.query(MachineSlot).first()


def _signed_trial(s, mould, good, total, idx):
    """Insert a historical settled+signed trial, exactly like the demo seed."""
    code = f'QE-H-{mould.code}-{idx:02d}'
    if s.query(Estimate).filter_by(code=code).first():
        return
    hours = 8.0
    e = Estimate(code=code, mould_id=mould.id, slot_id=_slot(s).id, rate=300,
                 setup_hours=1, predicted_hours=hours, quote_amount=5000,
                 status='committed', model_result={'note': 'history'})
    s.add(e)
    s.flush()
    started = (datetime(2026, 9, 1) + timedelta(days=idx)).isoformat()
    x = TrialExecution(estimate_id=e.id, started_at=started,
                       ended_at=(datetime(2026, 9, 1) + timedelta(days=idx, hours=hours)).isoformat(),
                       actual_hours=hours, good_shots=good, total_shots=total,
                       snapshot={'estimate_code': e.code, 'predicted_hours': hours,
                                 'quoted_amount': 5000, 'rate': 300,
                                 'mould_code': mould.code, 'slot_id': _slot(s).id},
                       status='settled')
    s.add(x)
    s.flush()
    s.add(CostReview(execution_id=x.id, material_cost=0, actual_cost=2400,
                     quoted_amount=5000, variance_hours=0, variance_pct=0,
                     margin=2600, reason='历史签结样本', status='signed',
                     signed_by=ADMIN.email))


def _committed_estimate(s, mould, code):
    if s.query(Estimate).filter_by(code=code).first():
        return s.query(Estimate).filter_by(code=code).one()
    e = Estimate(code=code, mould_id=mould.id, slot_id=_slot(s).id, rate=300,
                 setup_hours=1, predicted_hours=10, quote_amount=5000,
                 status='committed', model_result={'note': 'flow'})
    s.add(e)
    s.flush()
    return e


# ---------- 1. mould-grouped training & validation ----------

def test_train_per_mould_with_grouped_validation():
    with Session() as s:
        m_a, m_b = _mould(s, 'MD-A'), _mould(s, 'MD-B')
        # Six signed trials per mould; rates around 90% for MD-A.
        for i, good in enumerate([108, 112, 105, 115, 110, 107], start=1):
            _signed_trial(s, m_a, good, 120, i)
        for i, good in enumerate([90, 96], start=1):
            _signed_trial(s, m_b, good, 120, i)
        s.commit()

        out = booking.train_models(s, TrainInput(mould_id=m_a.id), ADMIN)
        s.commit()
        assert len(out['trained']) == 1 and out['failed'] == []
        m = out['trained'][0]
        assert m['sample_size'] == 6
        assert 0 < m['lower_rate'] <= m['mean_rate'] <= 1
        assert len(m['fold_rows']) == 6
        assert all(set(f) >= {'held_out_rate', 'loo_lower_rate', 'covered'} for f in m['fold_rows'])
        assert m['metrics']['loo_total'] == 6
        assert m['metrics']['loo_covered'] <= 6

        # Under-sampled mould must be refused, no model row written.
        out_b = booking.train_models(s, TrainInput(mould_id=m_b.id), ADMIN)
        s.commit()
        assert out_b['trained'] == [] and len(out_b['failed']) == 1
        assert '少于' in out_b['failed'][0]['reason']
        assert booking.latest_model(s, m_b.id) is None


def test_retrain_creates_new_version():
    with Session() as s:
        m_a = _mould(s, 'MD-A')
        booking.train_models(s, TrainInput(mould_id=m_a.id, z=2.0), ADMIN)
        s.commit()
        versions = [m.version for m in s.query(YieldModel).filter_by(mould_id=m_a.id)]
        assert versions == [1, 2]
        latest = booking.latest_model(s, m_a.id)
        assert latest.z == 2.0


# ---------- 2. ordering: commitment, suspension, over-occupancy ----------

def test_order_committed_by_bound_and_under_sampled_suspended():
    with Session() as s:
        m_a, m_b = _mould(s, 'MD-A'), _mould(s, 'MD-B')
        e_a = _committed_estimate(s, m_a, 'QE-FLOW-A')
        e_b = _committed_estimate(s, m_b, 'QE-FLOW-B')
        s.commit()

        model = booking.latest_model(s, m_a.id)
        o = booking.create_order(s, OrderInput(
            order_no='SO-001', customer='客户甲', estimate_id=e_a.id,
            planned_shots=100, requested_qty=50), EST)
        s.commit()
        expected = int(100 * model.lower_rate)
        assert o['status'] == 'committed'
        assert o['promised_qty'] == 50
        assert o['reserved_qty'] == 50
        assert o['rate_snapshot'] == model.lower_rate

        # Request below the bound: only the requested amount is promised.
        o2 = booking.create_order(s, OrderInput(
            order_no='SO-002', customer='客户乙', estimate_id=e_a.id,
            planned_shots=100, requested_qty=5), EST)
        s.commit()
        assert o2['promised_qty'] == 5

        # A third order would cross the mould/trial bound and is rejected.
        with pytest.raises(HTTPException) as exc:
            booking.create_order(s, OrderInput(
                order_no='SO-003', customer='客户丁', estimate_id=e_a.id,
                planned_shots=100, requested_qty=expected), EST)
        assert exc.value.status_code == 409 and '超占' in exc.value.detail

        # No trained bound -> suspended, nothing reserved.
        p = booking.create_order(s, OrderInput(
            order_no='SO-PEND', customer='客户丙', estimate_id=e_b.id,
            planned_shots=100, requested_qty=40), EST)
        s.commit()
        assert p['status'] == 'pending'
        assert p['promised_qty'] == 0 and p['reserved_qty'] == 0


def test_concurrent_orders_cannot_over_occupy_same_pool():
    with Session() as s:
        m_a = _mould(s, 'MD-A')
        e = _committed_estimate(s, m_a, 'QE-FLOW-A2')
        model = booking.latest_model(s, m_a.id)
        bound_qty = int(100 * model.lower_rate)
        first = 40
        assert first + 1 > bound_qty or bound_qty >= first
        booking.create_order(s, OrderInput(
            order_no='SO-OCC-1', customer='客户甲', estimate_id=e.id,
            planned_shots=100, requested_qty=first), EST)
        s.commit()
        with pytest.raises(HTTPException) as exc:
            booking.create_order(s, OrderInput(
                order_no='SO-OCC-2', customer='客户乙', estimate_id=e.id,
                planned_shots=100, requested_qty=bound_qty - first + 1), EST)
        assert exc.value.status_code == 409 and '超占' in exc.value.detail
        # Nothing was reserved by the rejected order.
        rows = s.query(SampleOrder).filter_by(estimate_id=e.id).all()
        assert sum(r.reserved_qty for r in rows) == first


# ---------- 3. completion lock: FIFO allocation & shortfall ----------

def test_completion_lock_allocates_fifo_and_books_shortfall():
    with Session() as s:
        m_a = _mould(s, 'MD-A')
        e = _committed_estimate(s, m_a, 'QE-LOCK')
        model = booking.latest_model(s, m_a.id)
        # Three orders, each promises 10, pool bound must allow 30.
        assert int(100 * model.lower_rate) >= 30
        for k in range(3):
            booking.create_order(s, OrderInput(
                order_no=f'SO-LOCK-{k}', customer=f'客户{k}', estimate_id=e.id,
                planned_shots=100, requested_qty=10), EST)
        s.commit()

        # Run the real floor flow: begin -> finish triggers the lock hook.
        start = datetime(2026, 11, 19, 8, 0)
        exec_svc.begin(s, StartTrial(estimate_id=e.id, started_at=start), EST)
        s.commit()
        x = s.query(TrialExecution).filter_by(estimate_id=e.id).one()
        # Only 23 good shots: first two orders fully covered, third gets 3.
        exec_svc.finish(s, x.id, FinishTrial(
            ended_at=start + timedelta(hours=9), good_shots=23, total_shots=30,
            overtime_reason=''), EST)
        s.commit()

        orders = s.query(SampleOrder).filter_by(estimate_id=e.id).order_by(SampleOrder.id).all()
        assert [o.order_no for o in orders] == ['SO-LOCK-0', 'SO-LOCK-1', 'SO-LOCK-2']
        assert orders[0].reserved_qty == 10 and orders[0].shortfall_qty == 0
        assert orders[0].status == 'locked'
        assert orders[1].reserved_qty == 10 and orders[1].status == 'locked'
        assert orders[2].reserved_qty == 3 and orders[2].shortfall_qty == 7
        assert orders[2].status == 'locked_short'
        assert all(o.locked_at for o in orders)
        assert sum(o.reserved_qty for o in orders) == 23

        # Locked list is immutable: a new order after completion is refused.
        with pytest.raises(HTTPException) as exc:
            booking.create_order(s, OrderInput(
                order_no='SO-LATE', customer='迟到客户', estimate_id=e.id,
                planned_shots=100, requested_qty=1), EST)
        assert exc.value.status_code == 409 and '锁定' in exc.value.detail


# ---------- 4. dispatch: sign gating, idempotent retry, no double deduct ----------

def test_dispatch_requires_settlement_and_retries_are_idempotent():
    with Session() as s:
        e = s.query(Estimate).filter_by(code='QE-LOCK').one()
        x = s.query(TrialExecution).filter_by(estimate_id=e.id).one()
        o1 = s.query(SampleOrder).filter_by(order_no='SO-LOCK-0').one()

        # Before signing the cost review, dispatch is refused.
        with pytest.raises(HTTPException) as exc:
            booking.dispatch(s, o1.id, BatchInput(batch_no='B1', qty=4), EST)
        assert exc.value.status_code == 409 and '签结' in exc.value.detail

        exec_svc.prepare_review(s, ReviewInput(execution_id=x.id, material_cost=0, reason=''), ADMIN)
        s.commit()
        review = s.query(CostReview).filter_by(execution_id=x.id).one()
        exec_svc.sign(s, review.id, ADMIN)
        s.commit()

        # First batch deducts 4.
        _, b1, replayed = booking.dispatch(s, o1.id, BatchInput(batch_no='B1', qty=4), EST)
        s.commit()
        assert replayed is False and b1['qty'] == 4
        o1 = s.get(SampleOrder, o1.id)
        assert o1.reserved_qty == 6 and o1.dispatched_qty == 4

        # Retry with the same batch number: no second deduction.
        _, b1_again, replayed = booking.dispatch(s, o1.id, BatchInput(batch_no='B1', qty=4), EST)
        s.commit()
        assert replayed is True and b1_again['id'] == b1['id']
        o1 = s.get(SampleOrder, o1.id)
        assert o1.reserved_qty == 6 and o1.dispatched_qty == 4
        assert s.query(SampleDispatch).filter_by(order_id=o1.id).count() == 1

        # A batch larger than the remaining reservation is rejected without deducting.
        with pytest.raises(HTTPException) as exc:
            booking.dispatch(s, o1.id, BatchInput(batch_no='B2', qty=7), EST)
        assert exc.value.status_code == 409 and '超过待发预留' in exc.value.detail
        o1 = s.get(SampleOrder, o1.id)
        assert o1.reserved_qty == 6 and o1.dispatched_qty == 4

        # Drain the order -> completed.
        booking.dispatch(s, o1.id, BatchInput(batch_no='B2', qty=6), EST)
        s.commit()
        o1 = s.get(SampleOrder, o1.id)
        assert o1.status == 'completed' and o1.reserved_qty == 0 and o1.dispatched_qty == 10

        # Receipt, then duplicate receipt refused.
        b = s.query(SampleDispatch).filter_by(batch_no='B1').one()
        booking.receive(s, b.id, ReceiptInput(tracking='RM-1', received_by='华远-王'), EST)
        s.commit()
        with pytest.raises(HTTPException) as exc:
            booking.receive(s, b.id, ReceiptInput(tracking='RM-1', received_by='华远-王'), EST)
        assert exc.value.status_code == 409 and '重复签收' in exc.value.detail


# ---------- 5. legacy deliveries keep counting, shared pool never over-sold ----------

def test_legacy_delivery_counts_against_shared_goods_pool():
    with Session() as s:
        e = s.query(Estimate).filter_by(code='QE-LOCK').one()
        x = s.query(TrialExecution).filter_by(estimate_id=e.id).one()
        review = s.query(CostReview).filter_by(execution_id=x.id).one()
        # Orders still holding reservations: order2 (10) + order3 (3) = 13;
        # order1 fully dispatched (10); total new-flow claim = 23 of 23 goods.
        free = booking.available_goods(s, x)
        assert free == 0
        with pytest.raises(HTTPException) as exc:
            from src import workshop_service
            workshop_service.deliver(s, DeliveryInput(
                review_id=review.id, good_samples=1, tracking='LEGACY-1'), EST)
        assert exc.value.status_code == 409 and '良品池余量' in exc.value.detail
        # No legacy delivery row was created by the failed attempt.
        assert s.query(TrialDelivery).join(
            CostReview, TrialDelivery.review_id == CostReview.id
        ).filter(CostReview.execution_id == x.id).count() == 0


def test_legacy_and_booking_share_pool_on_fresh_trial():
    with Session() as s:
        m_a = _mould(s, 'MD-A')
        e = _committed_estimate(s, m_a, 'QE-SHARE')
        booking.create_order(s, OrderInput(
            order_no='SO-SHARE-1', customer='客户甲', estimate_id=e.id,
            planned_shots=100, requested_qty=10), EST)
        s.commit()
        start = datetime(2026, 11, 20, 8, 0)
        exec_svc.begin(s, StartTrial(estimate_id=e.id, started_at=start), EST)
        s.commit()
        x = s.query(TrialExecution).filter_by(estimate_id=e.id).one()
        exec_svc.finish(s, x.id, FinishTrial(
            ended_at=start + timedelta(hours=9), good_shots=15, total_shots=20,
            overtime_reason=''), EST)
        exec_svc.prepare_review(s, ReviewInput(execution_id=x.id, material_cost=0, reason=''), ADMIN)
        review = s.query(CostReview).filter_by(execution_id=x.id).one()
        exec_svc.sign(s, review.id, ADMIN)
        s.commit()
        # 15 goods, booking reserves 10 -> 5 left for the legacy flow.
        from src import workshop_service
        workshop_service.deliver(s, DeliveryInput(
            review_id=review.id, good_samples=5, tracking='LEGACY-SHARE'), EST)
        s.commit()
        assert booking.available_goods(s, x) == 0
        o = s.query(SampleOrder).filter_by(order_no='SO-SHARE-1').one()
        with pytest.raises(HTTPException):
            booking.dispatch(s, o.id, BatchInput(batch_no='BX', qty=11), EST)
        # The reserved 10 can still go out.
        booking.dispatch(s, o.id, BatchInput(batch_no='BX', qty=10), EST)
        s.commit()


# ---------- 6. suspended orders released after the mould gains samples ----------

def test_pending_order_released_after_retraining():
    with Session() as s:
        m_b = _mould(s, 'MD-B')
        for i, good in enumerate([100, 104, 98, 110], start=3):
            _signed_trial(s, m_b, good, 120, i)
        s.commit()
        out = booking.train_models(s, TrainInput(mould_id=m_b.id), ADMIN)
        s.commit()
        assert len(out['trained']) == 1
        booking.evaluate_pending(s, ADMIN)
        s.commit()
        order = s.query(SampleOrder).filter_by(order_no='SO-PEND').one()
        assert order.status == 'committed'
        assert order.reserved_qty > 0


def test_shortfall_report_groups_gaps_and_suspensions():
    with Session() as s:
        report = booking.shortages(s)
        nos = {o['order_no'] for o in report['shortfall_orders']}
        pend = {o['order_no'] for o in report['suspended_orders']}
        assert 'SO-LOCK-2' in nos
        # SO-PEND was promoted, so nothing should remain suspended for it.
        assert 'SO-PEND' not in pend


# ---------- 7. HTTP smoke through the real FastAPI stack ----------

def test_http_stack_smoke():
    from src.main import app
    from src.auth import hash_password
    client = TestClient(app)
    with Session() as s:
        u = s.query(User).filter_by(email='estimator@ruimo.demo').one()
        u.password_hash = hash_password('Mould2026!')
        v = s.query(User).filter_by(email='viewer@ruimo.demo').one()
        v.password_hash = hash_password('Mould2026!')
        s.commit()
    login = client.post('/auth/login', json={'email': 'estimator@ruimo.demo',
                                             'password': 'Mould2026!'})
    assert login.status_code == 200
    headers = {'Authorization': 'Bearer ' + login.json()['token']}

    assert client.get('/yield-models', headers=headers).status_code == 200
    orders = client.get('/sample-orders', headers=headers).json()
    assert any(o['order_no'] == 'SO-001' for o in orders)
    gaps = client.get('/sample-shortfalls', headers=headers).json()
    assert 'shortfall_orders' in gaps and 'suspended_orders' in gaps
    enums = client.get('/enums', headers=headers).json()
    assert 'sample_order_states' in enums

    # Viewer cannot create orders.
    vlogin = client.post('/auth/login', json={'email': 'viewer@ruimo.demo',
                                              'password': 'Mould2026!'}).json()
    vh = {'Authorization': 'Bearer ' + vlogin['token']}
    m_b = None
    with Session() as s:
        m_b = _mould(s, 'MD-B')
        e = _committed_estimate(s, m_b, 'QE-HTTP')
        s.commit()
        eid = e.id
    resp = client.post('/sample-orders', headers=vh, json={
        'order_no': 'SO-HTTP-X', 'customer': '越权客户', 'estimate_id': eid,
        'planned_shots': 100, 'requested_qty': 5})
    assert resp.status_code == 403


def test_http_full_lifecycle_through_routes():
    from src.main import app
    from src.auth import hash_password
    client = TestClient(app)
    with Session() as s:
        u = s.query(User).filter_by(email='admin@ruimo.demo').one()
        u.password_hash = hash_password('Mould2026!')
        m_a = _mould(s, 'MD-A')
        e = _committed_estimate(s, m_a, 'QE-HTTP-LIFE')
        s.commit()
        eid = e.id
    tok = client.post('/auth/login', json={'email': 'admin@ruimo.demo',
                                           'password': 'Mould2026!'}).json()['token']
    h = {'Authorization': 'Bearer ' + tok}

    # Two orders before completion.
    r1 = client.post('/sample-orders', headers=h, json={
        'order_no': 'SO-HTTP-1', 'customer': '客户甲', 'estimate_id': eid,
        'planned_shots': 100, 'requested_qty': 30})
    assert r1.status_code == 200 and r1.json()['status'] == 'committed'
    r2 = client.post('/sample-orders', headers=h, json={
        'order_no': 'SO-HTTP-2', 'customer': '客户乙', 'estimate_id': eid,
        'planned_shots': 100, 'requested_qty': 30})
    assert r2.status_code == 200

    # Finish with only 40 good shots: FIFO leaves the second order short.
    start = datetime(2026, 11, 21, 8, 0)
    client.post('/executions', headers=h, json={
        'estimate_id': eid, 'started_at': start.isoformat()})
    with Session() as s:
        x = s.query(TrialExecution).filter_by(estimate_id=eid).one()
        xid = x.id
    client.post(f'/executions/{xid}/finish', headers=h, json={
        'ended_at': (start + timedelta(hours=8)).isoformat(),
        'good_shots': 40, 'total_shots': 50, 'overtime_reason': ''})
    with Session() as s:
        o1 = s.query(SampleOrder).filter_by(order_no='SO-HTTP-1').one()
        o2 = s.query(SampleOrder).filter_by(order_no='SO-HTTP-2').one()
        assert o1.status == 'locked' and o1.reserved_qty == 30
        assert o2.status == 'locked_short' and o2.reserved_qty == 10 and o2.shortfall_qty == 20

    # Sign the review, then dispatch in batches with idempotent retry.
    client.post('/cost-reviews', headers=h, json={
        'execution_id': xid, 'material_cost': 0, 'reason': ''})
    with Session() as s:
        rid = s.query(CostReview).filter_by(execution_id=xid).one().id
    client.post(f'/cost-reviews/{rid}/sign', headers=h)

    d1 = client.post(f'/sample-orders/{o1.id}/dispatches', headers=h,
                     json={'batch_no': 'HB-1', 'qty': 20})
    assert d1.status_code == 200 and d1.json()['idempotent_replay'] is False
    again = client.post(f'/sample-orders/{o1.id}/dispatches', headers=h,
                        json={'batch_no': 'HB-1', 'qty': 20})
    assert again.status_code == 200 and again.json()['idempotent_replay'] is True
    with Session() as s:
        o1 = s.get(SampleOrder, o1.id)
        assert o1.dispatched_qty == 20 and o1.reserved_qty == 10
        assert s.query(SampleDispatch).filter_by(order_id=o1.id).count() == 1

    # Over-dispatch is rejected without deducting.
    bad = client.post(f'/sample-orders/{o1.id}/dispatches', headers=h,
                      json={'batch_no': 'HB-2', 'qty': 11})
    assert bad.status_code == 409
    client.post(f'/sample-orders/{o1.id}/dispatches', headers=h,
                json={'batch_no': 'HB-2', 'qty': 10})
    with Session() as s:
        assert s.get(SampleOrder, o1.id).status == 'completed'

    # Receipt flow and duplicate-receipt rejection.
    with Session() as s:
        bid = s.query(SampleDispatch).filter_by(batch_no='HB-1').one().id
    assert client.post(f'/sample-dispatches/{bid}/receive', headers=h,
                       json={'tracking': 'TR-1', 'received_by': '客户甲-李'}).status_code == 200
    assert client.post(f'/sample-dispatches/{bid}/receive', headers=h,
                       json={'tracking': 'TR-1', 'received_by': '客户甲-李'}).status_code == 409

    # Shortfall board exposes the second order.
    gaps = client.get('/sample-shortfalls', headers=h).json()
    assert any(x['order_no'] == 'SO-HTTP-2' and x['shortfall_qty'] == 20
               for x in gaps['shortfall_orders'])
