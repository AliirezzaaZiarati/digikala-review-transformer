from pathlib import Path

# ==========================================
# EXCLUDE LIST
# نام فایل‌ها و پوشه‌هایی که نباید نمایش داده شوند
# ==========================================

EXCLUDE = [
    # "venv",
    "__pycache__",
    # "node_modules",
    # ".git",
]

# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path.cwd()
EXCLUDE_HIDDEN_CONTENTS = True  # پوشه‌های نقطه‌دار فقط با نام نمایش داده شوند


# ==========================================
# TREE GENERATOR
# ==========================================

def print_tree(directory: Path, prefix: str = ""):

    try:
        entries = sorted(
            directory.iterdir(),
            key=lambda p: (p.is_file(), p.name.lower())
        )
    except PermissionError:
        return

    # فیلتر کردن موارد Exclude
    entries = [
        item for item in entries
        if item.name not in EXCLUDE
    ]

    for index, item in enumerate(entries):

        is_last = index == len(entries) - 1

        connector = "└── " if is_last else "├── "
        print(prefix + connector + item.name)

        # اگر پوشه مخفی است، محتوای آن را نمایش نده
        if item.is_dir() and not (
            EXCLUDE_HIDDEN_CONTENTS and item.name.startswith(".")
        ):
            extension = "    " if is_last else "│   "
            print_tree(item, prefix + extension)


# ==========================================
# RUN
# ==========================================

print(ROOT.name + "/")
print_tree(ROOT)
