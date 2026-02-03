from pathlib import Path

print(Path(__file__).resolve().parents[1])

print(Path(__file__).resolve().parent.parent)