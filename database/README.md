# Database Configuration

## Supabase / PostgreSQL

1. Create a new Supabase project.
2. Run the SQL in `schema.sql` via the Supabase SQL editor or `psql`:
   ```sql
   \i backend/database/schema.sql
   ```
3. Note the `SUPABASE_URL` and `SUPABASE_KEY` values and add them to `backend/.env`.

## Pinecone

1. Export your Pinecone API key & environment in `backend/.env`.
2. Run the provisioning helper:
   ```bash
   cd backend
   python -m database.pinecone_setup
   ```
3. The script creates a 768-dim cosine index ready for hybrid search metadata.
