-- 0006_optimize_auth_rls_policies.sql
-- Wrap auth.uid() calls in SELECT so Postgres can initplan them once per query.

drop policy if exists "profiles are viewable by owner" on public.profiles;
drop policy if exists "profiles are updatable by owner" on public.profiles;
drop policy if exists "scans are viewable by owner" on public.scans;
drop policy if exists "scans are insertable by owner" on public.scans;

create policy "profiles are viewable by owner"
    on public.profiles for select
    using ((select auth.uid()) = id);

create policy "profiles are updatable by owner"
    on public.profiles for update
    using ((select auth.uid()) = id);

create policy "scans are viewable by owner"
    on public.scans for select
    using ((select auth.uid()) = user_id);

create policy "scans are insertable by owner"
    on public.scans for insert
    with check ((select auth.uid()) = user_id);
