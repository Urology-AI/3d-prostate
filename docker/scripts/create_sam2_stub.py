"""Create SAM2 stub so monailabel doesn't crash on import"""
import site, os

sp = site.getsitepackages()[0]
os.makedirs(f"{sp}/sam2", exist_ok=True)

with open(f"{sp}/sam2/__init__.py", "w") as f:
    f.write('__version__ = "1.1.0"\n')

dist = f"{sp}/sam2-1.1.0.dist-info"
os.makedirs(dist, exist_ok=True)

with open(f"{dist}/METADATA", "w") as f:
    f.write("Metadata-Version: 2.1\nName: sam2\nVersion: 1.1.0\n")
with open(f"{dist}/INSTALLER", "w") as f:
    f.write("pip\n")
with open(f"{dist}/WHEEL", "w") as f:
    f.write("Wheel-Version: 1.0\nGenerator: stub\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
with open(f"{dist}/RECORD", "w") as f:
    f.write("sam2/__init__.py,sha256=abc123,20\n")

print(f"SAM2 stub created at {sp}")
