from fastapi import APIRouter, Depends
from sqlalchemy import select
from .db import db_session
from .auth import current, writers
from .services import serialize
from .booking_models import SampleOrder, SampleDispatch
from .booking_schemas import OrderInput, BatchInput, ReceiptInput, TrainInput
from . import booking_service as svc

router = APIRouter()


@router.get('/yield-models')
def yield_models(s=Depends(db_session), u=Depends(current)):
    return svc.yield_models(s)


@router.post('/yield-models/train')
def train(p: TrainInput, s=Depends(db_session), u=Depends(writers)):
    out = svc.train_models(s, p, u)
    s.commit()
    return out


@router.post('/sample-orders/evaluate')
def evaluate(s=Depends(db_session), u=Depends(writers)):
    svc.evaluate_pending(s, u)
    s.commit()
    return svc.shortages(s)


@router.get('/sample-orders')
def orders(s=Depends(db_session), u=Depends(current)):
    return [svc.view(s, o) for o in
            s.scalars(select(SampleOrder).order_by(SampleOrder.id.desc()))]


@router.post('/sample-orders')
def create_order(p: OrderInput, s=Depends(db_session), u=Depends(writers)):
    o = svc.create_order(s, p, u)
    s.commit()
    return o


@router.get('/sample-shortfalls')
def shortfalls(s=Depends(db_session), u=Depends(current)):
    return svc.shortages(s)


@router.get('/sample-dispatches')
def dispatches(s=Depends(db_session), u=Depends(current)):
    return [serialize(b) for b in
            s.scalars(select(SampleDispatch).order_by(SampleDispatch.id.desc()))]


@router.post('/sample-orders/{oid}/dispatches')
def dispatch(oid: int, p: BatchInput, s=Depends(db_session), u=Depends(writers)):
    order, batch, replayed = svc.dispatch(s, oid, p, u)
    s.commit()
    return {'order': order, 'batch': batch, 'idempotent_replay': replayed}


@router.post('/sample-dispatches/{bid}/receive')
def receive(bid: int, p: ReceiptInput, s=Depends(db_session), u=Depends(writers)):
    b = svc.receive(s, bid, p, u)
    s.commit()
    return b
