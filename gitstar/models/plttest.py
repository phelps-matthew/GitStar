import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset"
IMG_PATH = BASE_DIR / "features"
IMG_PATH.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots()
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
fig.savefig(
    str(IMG_PATH / "full_seaborn/log_canonical_kde.png"),
    transparent=False,
    dpi=300,
    bbox_inches="tight",
)
