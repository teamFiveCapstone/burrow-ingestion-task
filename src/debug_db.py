import psycopg2

DB_HOST = "burrow-zach.cluster-cwxgyacqyoae.us-east-1.rds.amazonaws.com"
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "password"  # dev only

# This is what PGVectorStore will actually create for table_name="burrow_table"
REAL_TABLE = "data_burrow_table"


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

    # List public tables
    print("\n== Public tables ==")
    cur.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
    )
    for (table_name,) in cur.fetchall():
        print(" -", table_name)

    # Show schema for the vector table
    print(f"\n== {REAL_TABLE} schema ==")
    cur.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s;
        """,
        (REAL_TABLE,),
    )
    rows = cur.fetchall()
    if not rows:
        print(f"Table {REAL_TABLE} not found.")
    else:
        for col, dtype in rows:
            print(f" {col}: {dtype}")

    # Count rows in the vector table
    print(f"\n== {REAL_TABLE} row count ==")
    try:
        cur.execute(f"SELECT COUNT(*) FROM {REAL_TABLE};")
        count = cur.fetchone()[0]
        print(f"Rows in {REAL_TABLE}: {count}")
    except Exception as e:
        print(f"Could not query {REAL_TABLE}: {e}")

    cur.close()
    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
