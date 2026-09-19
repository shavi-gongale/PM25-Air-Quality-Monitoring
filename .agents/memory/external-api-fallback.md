---
name: External API fallback
description: Behavior for optional weather and air-quality provider credentials.
---

Live provider APIs should remain the primary source, but missing credentials and temporary provider failures must produce labeled demo data rather than `None` values that break combined workflows.

**Why:** The project intentionally supports demonstration mode, and one provider already used that pattern while the current-weather path still failed hard when its secret was absent.

**How to apply:** Mark fallback readings as non-live, show a clear configuration message, and never persist fallback readings as observed measurements.