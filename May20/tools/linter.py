import asyncio
import tempfile
import os

async def lint_code(code: str) -> str:
    # Create a temporary file to save the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp:
        temp.write(code)
        temp_path = temp.name

    try:
        # Run pylint as an asynchronous subprocess
        process = await asyncio.create_subprocess_exec(
            'pylint', temp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        # Combine and return outputs
        output = stdout.decode().strip()
        error = stderr.decode().strip()

        return f"Pylint Output:\n{output}\n\nErrors:\n{error}" if error else f"Pylint Output:\n{output}"

    except FileNotFoundError:
        return "❌ Error: 'pylint' not found. Please install it using `pip install pylint`."
    except Exception as e:
        return f"❌ An error occurred while linting: {str(e)}"
    finally:
        # Clean up the temp file
        os.remove(temp_path)