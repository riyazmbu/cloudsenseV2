from typing import Optional
from pydantic import BaseModel, ConfigDict

class BillingRecordResponse(BaseModel):
    id: int
    provider: str
    billing_month: Optional[str] = None
    service: Optional[str] = None
    region: Optional[str] = None
    resource_id: Optional[str] = None
    environment: Optional[str] = None
    team: Optional[str] = None
    daily_cost: float
    monthly_cost: float
    cpu_utilization: Optional[float] = None
    memory_utilization: Optional[float] = None
    hours_running: Optional[float] = None
    source_file: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class BillingDocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    extraction_method: Optional[str] = None
    ocr_required: bool
    extracted_text: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
