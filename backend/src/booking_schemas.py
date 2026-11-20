from pydantic import BaseModel, Field, ConfigDict

class Clean(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

class OrderInput(Clean):
    order_no: str = Field(min_length=2, max_length=60)
    customer: str = Field(min_length=2, max_length=120)
    estimate_id: int = Field(gt=0)
    planned_shots: int = Field(gt=0, le=10_000_000)
    requested_qty: int = Field(gt=0, le=1_000_000)
    note: str = Field(default='', max_length=600)

class BatchInput(Clean):
    batch_no: str = Field(min_length=2, max_length=60)
    qty: int = Field(gt=0, le=1_000_000)

class ReceiptInput(Clean):
    tracking: str = Field(min_length=2, max_length=120)
    received_by: str = Field(min_length=2, max_length=120)

class TrainInput(Clean):
    mould_id: int | None = Field(default=None, gt=0)
    z: float = Field(default=1.645, gt=0, le=4)
