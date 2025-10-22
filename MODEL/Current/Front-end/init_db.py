from app import app, db

# Reset database with new schema (no email column)
with app.app_context():
    print("Resetting database to remove email column...")

    # Drop the user table if it exists
    try:
        db.session.execute(db.text('DROP TABLE IF EXISTS "user"'))
        db.session.commit()
        print("✓ Dropped existing user table")
    except Exception as e:
        print(f"✗ Error dropping table: {e}")

    # Create all tables with new schema
    try:
        db.create_all()
        print("✓ Database tables created successfully with new schema!")
    except Exception as e:
        print(f"✗ Error creating tables: {e}")

    # Test the connection
    try:
        result = db.session.execute(db.text('SELECT name FROM sqlite_master WHERE type="table"')).fetchall()
        print(f"✓ Available tables: {[row[0] for row in result]}")

        # Check user table structure
        user_columns = db.session.execute(db.text('PRAGMA table_info("user")')).fetchall()
        print("✓ User table columns:")
        for col in user_columns:
            print(f"  - {col[1]} ({col[2]}) {'NOT NULL' if col[3] else 'NULL'}")

    except Exception as e:
        print(f"✗ Database test failed: {e}")
