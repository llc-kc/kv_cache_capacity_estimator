# SPDX-License-Identifier: Apache-2.0
"""Defaults shared by the public API, JSON config, and command line."""

DEFAULT_PAGE_SIZE = 64
DEFAULT_CAPACITY_LABELS = (
    "100GiB",
    "200GiB",
    "400GiB",
    "800GiB",
    "1TiB",
    "2TiB",
    "4TiB",
    "6TiB",
    "8TiB",
    "12TiB",
    "16TiB",
    "24TiB",
    "32TiB",
    "64TiB",
)
DEFAULT_CAPACITIES = ",".join(DEFAULT_CAPACITY_LABELS)
