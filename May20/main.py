import asyncio

from agents.coder import Coder
from agents.debugger import Debugger
from tools.python_executor import execute_python_code
from tools.linter import lint_code

coder = Coder()
debugger = Debugger()

async def main():
    while True:
        user_input = input("\nPrompt: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        code = await coder.respond(user_input)
        print("\n🧠 Coder \n```python")
        print(code.strip())
        print("```")

        result = await execute_python_code(code)
        print("\n🐞 Debugger [PythonExecutor]")
        print("✅ Execution Output:\n", result)

        lint_result = await lint_code(code)
        print("\n🧹 PylintLinter")
        print("📋 Pylint Linter Output:\n", lint_result)

if __name__ == "__main__":
    asyncio.run(main())