"""Patch monailabel config.py to handle None package names in distributions()"""
import sys

CONFIG = "/usr/local/lib/python3.10/dist-packages/monailabel/config.py"

try:
    with open(CONFIG, "r") as f:
        content = f.read()

    OLD = "return name in sorted(x.name for x in distributions())"
    NEW = "return name in sorted(x.name for x in distributions() if x.name is not None)"

    if OLD in content:
        content = content.replace(OLD, NEW)
        with open(CONFIG, "w") as f:
            f.write(content)
        print("Patched monailabel/config.py successfully")
    else:
        print("Pattern not found — may already be patched or different version")
        for i, line in enumerate(content.splitlines()):
            if "distributions" in line:
                print(f"  Line {i+1}: {line.strip()}")

except FileNotFoundError:
    # CPU image uses different python path
    import glob
    configs = glob.glob("/usr/local/lib/python*/dist-packages/monailabel/config.py")
    if not configs:
        configs = glob.glob("/usr/lib/python*/dist-packages/monailabel/config.py")

    for config in configs:
        with open(config, "r") as f:
            content = f.read()
        OLD = "return name in sorted(x.name for x in distributions())"
        NEW = "return name in sorted(x.name for x in distributions() if x.name is not None)"
        if OLD in content:
            content = content.replace(OLD, NEW)
            with open(config, "w") as f:
                f.write(content)
            print(f"Patched {config}")
