import psycopg2

DB_HOST = "burrow-zach.cluster-cwxgyacqyoae.us-east-1.rds.amazonaws.com"
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "password"  # dev only


def main():
    print(f"Connecting to {DB_HOST}:{DB_PORT}/{DB_NAME} ...")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=60,
    )
    conn.autocommit = True
    cur = conn.cursor()

    # ======================================================
    # 1. LIST ALL TABLES IN *PUBLIC* SCHEMA
    # ======================================================
    print("\n== TABLES IN 'public' SCHEMA ==")
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)

    tables = [row[0] for row in cur.fetchall()]
    for t in tables:
        print(" -", t)

    # ======================================================
    # 2. SELECT * FROM EACH TABLE
    # ======================================================
    print("\n== TABLE CONTENTS ==")

    for table in tables:
        print(f"\n----- {table} -----")

        try:
            cur.execute(f"SELECT * FROM {table} LIMIT 50;")  # safety: limit 50 rows
            rows = cur.fetchall()

            if not rows:
                print("(empty)")
            else:
                for row in rows:
                    print(row)

        except Exception as e:
            print(f"Error reading {table}: {e}")

    cur.close()
    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
