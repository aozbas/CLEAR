-- 0005_lock_rls_auto_enable.sql
-- Some Supabase projects may have this helper event-trigger function in public.
-- It should not be callable through PostgREST by anon or authenticated users.

do $$
begin
    if to_regprocedure('public.rls_auto_enable()') is not null then
        revoke execute on function public.rls_auto_enable() from public;
        revoke execute on function public.rls_auto_enable() from anon, authenticated;
    end if;
end;
$$;
