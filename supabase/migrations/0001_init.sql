-- 0001_init.sql
-- Initial schema: profiles + scans, with RLS so users only see their own data.

create table if not exists public.profiles (
    id          uuid primary key references auth.users(id) on delete cascade,
    email       text,
    created_at  timestamptz not null default now()
);

create table if not exists public.scans (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references public.profiles(id) on delete cascade,
    image_url   text not null,
    prediction  text,
    confidence  real,
    created_at  timestamptz not null default now()
);

create index if not exists scans_user_id_created_at_idx
    on public.scans (user_id, created_at desc);

-- Row-Level Security
alter table public.profiles enable row level security;
alter table public.scans    enable row level security;

create policy "profiles are viewable by owner"
    on public.profiles for select
    using (auth.uid() = id);

create policy "profiles are updatable by owner"
    on public.profiles for update
    using (auth.uid() = id);

create policy "scans are viewable by owner"
    on public.scans for select
    using (auth.uid() = user_id);

create policy "scans are insertable by owner"
    on public.scans for insert
    with check (auth.uid() = user_id);
