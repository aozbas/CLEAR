-- 0007_add_scan_model_version.sql
-- Record which experimental model produced each saved scan result.

alter table public.scans
    add column if not exists model_version text;
