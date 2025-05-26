from datetime import datetime
from chrom_db.config import CONV_FILE

def log(text: str):
    ts = datetime.now().isoformat()
    with open(CONV_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {text}\n")
