-- schema.sql references 'service_role' in its RLS policy, which is a Supabase-specific
-- role that doesn't exist in vanilla PostgreSQL. Create it before schema.sql runs.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'service_role') THEN
        CREATE ROLE service_role NOLOGIN;
    END IF;
END;
$$;

-- Grant the app user BYPASSRLS so it can insert/select despite RLS being enabled.
-- In Supabase the service_role key implicitly bypasses RLS; we replicate that here.
ALTER ROLE salus BYPASSRLS;
