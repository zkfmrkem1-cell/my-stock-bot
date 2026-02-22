from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class DatasetExport(Base):
    __tablename__ = "dataset_exports"
    __table_args__ = (
        sa.UniqueConstraint("dataset_name", "dataset_version", name="uq_exp_dataset_exports_name_ver"),
        sa.Index("ix_exp_dataset_exports_as_of_date", "as_of_date"),
        {"schema": "exp"},
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    dataset_name: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    dataset_version: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    as_of_date: Mapped[date] = mapped_column(sa.Date, nullable=False)
    row_count: Mapped[int | None] = mapped_column(sa.Integer)
    status: Mapped[str] = mapped_column(
        sa.String(24),
        nullable=False,
        server_default=sa.text("'ready'"),
    )
    storage_uri: Mapped[str | None] = mapped_column(sa.Text)
    split_meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    export_meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    )

