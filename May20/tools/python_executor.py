import asyncio
import sys
import io
from typing import Any

async def execute_python_code(code: str) -> dict[str, Any]:
    loop = asyncio.get_event_loop()
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        exec_globals = {}
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout, stderr
        await loop.run_in_executor(None, exec, code, exec_globals)
    except Exception as e:
        return {"success": False, "error": str(e), "output": stderr.getvalue()}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return {"success": True, "output": stdout.getvalue()}