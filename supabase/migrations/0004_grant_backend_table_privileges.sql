-- 0004_grant_backend_table_privileges.sql
-- Allow the trusted FastAPI backend, which uses the Supabase service-role key,
-- to save and read Phase 1 scan results through PostgREST.

grant usage on schema public to service_role;
grant select, insert on table public.scans to service_role;
