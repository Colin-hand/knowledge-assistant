"""Sub-document access markers, shared by chunker (isolation) and enricher (ACL).

Assumption: this marker convention is how sub-document restrictions are
expressed in this corpus (case-insensitive, tolerant of em/en/hyphen dashes).
Known instance: one paragraph in general/all-hands-2025-q2.pdf.
"""

import re

EXEC_MARKER = re.compile(r"confidential\s*[—–-]+\s*executive\s+committee\s+only", re.IGNORECASE)
