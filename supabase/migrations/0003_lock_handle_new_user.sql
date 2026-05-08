-- 0003_lock_handle_new_user.sql
-- Revoke client-facing EXECUTE on handle_new_user.
--
-- Why: handle_new_user is SECURITY DEFINER and lives in `public`, which means
-- PostgREST exposes it as POST /rest/v1/rpc/handle_new_user with a default
-- EXECUTE grant to PUBLIC (anon + authenticated). Calling it returns an error
-- because it expects a trigger context, but defense in depth says don't expose
-- definer functions to clients at all. The on_auth_user_created trigger keeps
-- working — triggers run as the table owner, not the caller.

revoke execute on function public.handle_new_user() from public;
revoke execute on function public.handle_new_user() from anon, authenticated;
