"""Pre-completion sample reservations: mould-level good-rate lower bounds,
per-order commitments, completion locks and idempotent sample dispatches."""
from sqlalchemy import String, Integer, Float, ForeignKey, JSON, CheckConstraint, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from .db import Base

class YieldModel(Base):
    """Good-rate lower bound trained per mould from signed actuals."""
    __tablename__ = 'yield_models'
    __table_args__ = (
        UniqueConstraint('mould_id', 'version'),
        CheckConstraint('sample_size >= 0 AND 0 < lower_rate AND lower_rate <= 1'),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    mould_id: Mapped[int] = mapped_column(ForeignKey('moulds.id'))
    version: Mapped[int] = mapped_column(Integer, default=1)
    lower_rate: Mapped[float] = mapped_column(Float)
    sample_size: Mapped[int] = mapped_column(Integer)
    mean_rate: Mapped[float] = mapped_column(Float)
    std_rate: Mapped[float] = mapped_column(Float)
    z: Mapped[float] = mapped_column(Float)
    fold_rows: Mapped[list] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    trained_at: Mapped[str] = mapped_column(String(40))

class SampleOrder(Base):
    """A customer sample order placed before the trial is finished."""
    __tablename__ = 'sample_orders'
    __table_args__ = (
        UniqueConstraint('order_no'),
        CheckConstraint('requested_qty > 0 AND promised_qty >= 0 '
                        'AND promised_qty <= requested_qty '
                        'AND committed_qty >= 0 AND reserved_qty >= 0 '
                        'AND dispatched_qty >= 0 AND shortfall_qty >= 0 '
                        'AND reserved_qty + dispatched_qty <= committed_qty '
                        'AND committed_qty <= promised_qty '
                        'AND shortfall_qty <= promised_qty'),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    order_no: Mapped[str] = mapped_column(String(60), unique=True)
    customer: Mapped[str] = mapped_column(String(120))
    estimate_id: Mapped[int] = mapped_column(ForeignKey('estimates.id'))
    mould_id: Mapped[int] = mapped_column(ForeignKey('moulds.id'))
    planned_shots: Mapped[int] = mapped_column(Integer)
    requested_qty: Mapped[int] = mapped_column(Integer)
    promised_qty: Mapped[int] = mapped_column(Integer, default=0)
    committed_qty: Mapped[int] = mapped_column(Integer, default=0)
    reserved_qty: Mapped[int] = mapped_column(Integer, default=0)
    dispatched_qty: Mapped[int] = mapped_column(Integer, default=0)
    shortfall_qty: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default='pending')
    rate_snapshot: Mapped[float | None] = mapped_column(Float, nullable=True)
    locked_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    note: Mapped[str] = mapped_column(String(600), default='')

class SampleDispatch(Base):
    """One dispatch batch; the natural key makes retries idempotent."""
    __tablename__ = 'sample_dispatches'
    __table_args__ = (
        UniqueConstraint('order_id', 'batch_no'),
        CheckConstraint('qty > 0'),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey('sample_orders.id'))
    batch_no: Mapped[str] = mapped_column(String(60))
    qty: Mapped[int] = mapped_column(Integer)
    state: Mapped[str] = mapped_column(String(20), default='dispatched')
    snapshot: Mapped[dict] = mapped_column(JSON)
    tracking: Mapped[str] = mapped_column(String(120), default='')
    received_by: Mapped[str] = mapped_column(String(120), default='')
