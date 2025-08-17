"""add links table

Revision ID: add_links_0001
Revises: b1f989a4b3fc
Create Date: 2025-03-11 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_links_0001'
down_revision: Union[str, None] = 'b1f989a4b3fc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'links',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('short_code', sa.String(length=64), nullable=False),
        sa.Column('original_url', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('click_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_accessed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_links_short_code'), 'links', ['short_code'], unique=True)


def downgrade() -> None:
    op.drop_index(op.f('ix_links_short_code'), table_name='links')
    op.drop_table('links')


