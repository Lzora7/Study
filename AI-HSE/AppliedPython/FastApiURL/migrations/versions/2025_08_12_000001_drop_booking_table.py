"""drop booking table

Revision ID: drop_booking_0001
Revises: add_links_0001
Create Date: 2025-08-12 00:00:01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'drop_booking_0001'
down_revision: Union[str, None] = 'add_links_0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table('booking')


def downgrade() -> None:
    op.create_table(
        'booking',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('room_id', sa.Integer(), nullable=True),
        sa.Column('category', sa.String(), nullable=True),
        sa.Column('start_time', sa.DateTime(), nullable=False),
        sa.Column('end_time', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )


