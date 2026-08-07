"""Alembic environment, wired to the application's own configuration.

Two deliberate choices:

1. **The URL comes from `Settings`, not `alembic.ini`.** Migrations therefore
   target exactly the database the API talks to, and no credentials sit in a
   checked-in file. `ALEMBIC_DATABASE_URL` overrides it for the case where
   migrations run from a bastion with a different host than the app.

2. **Async engine.** The application uses `postgresql+asyncpg`; running
   migrations through a separate sync driver would mean the schema is created
   by a connection path nobody tests. `run_async_migrations` uses the same
   driver the app does.
"""

from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Make the `app` package importable when alembic is invoked from api/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.core.db import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Autogenerate compares against this metadata.
target_metadata = Base.metadata


def _database_url() -> str:
    return os.getenv("ALEMBIC_DATABASE_URL") or get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting.

    Used to hand a reviewable script to a DBA — in regulated environments the
    migration is often applied by someone other than the deployer.
    """
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Catch column type and default drift, not just added/removed columns.
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
