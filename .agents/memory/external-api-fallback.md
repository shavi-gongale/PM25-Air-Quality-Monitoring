---
name: External API fallback
description: Behavior for optional weather and air-quality provider credentials.
---

Live provider APIs must remain the only source for live-data features. Missing credentials and temporary provider failures should produce a clear unavailable state rather than fabricated readings.

**Why:** The application is intended to report monitoring data, and mock values can be mistaken for real observations or be persisted into model-training data.

**How to apply:** Keep provider credentials in Replit Secrets, log the provider's status/error, return an empty/unavailable result, and explain which secret is required in the UI.