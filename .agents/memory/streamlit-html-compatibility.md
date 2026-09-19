---
name: Streamlit HTML compatibility
description: Compatibility constraints for custom HTML rendered through Streamlit markdown.
---

Custom HTML passed through `st.markdown(..., unsafe_allow_html=True)` must remain passive markup. Inline JavaScript and event-handler attributes can be interpreted by Streamlit's HTML-to-React renderer as invalid string event props and crash the entire browser view.

**Why:** The app's browser runtime failed with React error 231 (`onClick` was a string) even though the Python workflow remained healthy.

**How to apply:** Use Streamlit widgets or ordinary links/query parameters for interaction. Keep decorative custom markup free of `<script>` blocks and `onclick`/similar event attributes.