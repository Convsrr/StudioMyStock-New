"""Mark any pending/running jobs as failed. Useful when the schema changed
or a code bug left jobs hanging. Run from the api/ directory."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def main() -> None:
    db = Path("storage/app.db")
    if not db.exists():
        print("no DB at storage/app.db; nothing to do")
        return
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(
        "UPDATE jobs SET status='failed', error_code='reset', "
        "error_message='reset by maintenance script' "
        "WHERE status IN ('pending','running')"
    )
    con.commit()
    print(f"updated {cur.rowcount} jobs")


if __name__ == "__main__":
    main()
