import streamlit.web.bootstrap
import os
import sys

# Set the environment variables needed
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
os.environ["STREAMLIT_SERVER_PORT"] = "5000"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# Run the streamlit app
streamlit.web.bootstrap.run("app.py", "", [], [])