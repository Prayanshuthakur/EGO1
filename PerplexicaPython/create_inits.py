"""
Initialize all __init__.py files for package structure.
"""

# Create empty __init__.py files for all packages
packages = [
    "llm",
    "search",
    "agents",
    "agents/researcher",
    "agents/researcher/actions",
    "widgets",
    "prompts",
    "ui",
    "ui/components",
    "tests"
]

for package in packages:
    with open(f"{package}/__init__.py", "w") as f:
        f.write(f'"""{package.split("/")[-1].capitalize()} package initialization."""\n')

print("Created all __init__.py files")
