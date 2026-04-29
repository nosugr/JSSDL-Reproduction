import mcp.server.fastmcp as fastmcp
import pymupdf4llm
import os

# Initialize FastMCP server
mcp = fastmcp.FastMCP("PDF Reader")

@mcp.tool()
def read_pdf(path: str) -> str:
    """Reads a PDF file and returns its content as Markdown.
    
    Args:
        path: The absolute path to the PDF file.
    """
    if not os.path.isfile(path):
        return f"Error: File '{path}' not found."
    
    try:
        # Convert PDF to Markdown using pymupdf4llm for high-quality output
        md_text = pymupdf4llm.to_markdown(path)
        return md_text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

if __name__ == "__main__":
    mcp.run()
