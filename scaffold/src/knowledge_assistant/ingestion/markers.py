"""Confidential-marker regex shared by chunker and enricher."""

import re

EXEC_MARKER = re.compile(r"confidential\s*[—–-]+\s*executive\s+committee\s+only", re.IGNORECASE)
