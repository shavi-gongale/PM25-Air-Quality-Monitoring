import base64
from io import BytesIO

def get_download_link(data, filename, mime_type):
    """
    Generate a download link for the given data.
    
    Args:
        data: The data to be downloaded (string or bytes)
        filename (str): The name of the file to be downloaded
        mime_type (str): The MIME type of the file
        
    Returns:
        str: HTML code for the download link
    """
    # Convert data to bytes if it's a string
    if isinstance(data, str):
        data = data.encode()
    
    # Encode the data as base64
    b64 = base64.b64encode(data).decode()
    
    # Generate the download link
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}" class="btn-download">Download {filename}</a>'
    
    # Add some style to the link
    styled_href = f"""
    <style>
    .btn-download {{
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #4CAF50;
        color: white;
        text-align: center;
        text-decoration: none;
        font-size: 16px;
        border-radius: 4px;
        cursor: pointer;
        margin: 0.5rem 0;
        border: none;
        transition: background-color 0.3s;
    }}
    .btn-download:hover {{
        background-color: #45a049;
    }}
    </style>
    {href}
    """
    
    return styled_href
