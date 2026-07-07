-- 0008_allow_scan_history_without_images.sql
-- Privacy-first scan history stores experimental prediction metadata without
-- requiring a persisted uploaded photo.

alter table public.scans
    alter column image_url drop not null;
