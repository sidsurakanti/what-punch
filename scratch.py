import os 
from pathlib import Path

cwd = Path.cwd() / "assets"
print(cwd, type(cwd))

for folder in cwd.iterdir():
    if folder.is_dir() and folder.name == "idle":
        files = sorted(folder.glob("*.png"))
        for idx, file in enumerate(files):
            name = f"{folder.name}_{idx:03}{file.suffix}"
            new_path = file.with_name(name) 
            os.rename(file, new_path)
        



