-- 0002_profile_trigger_and_constraints.sql
-- 1. Auto-create a profiles row whenever a new user signs up via Supabase Auth.
-- 2. Add schema-level guardrails on the scans table (confidence range, valid labels).

-- ============================================================
-- 1. Profile auto-creation trigger
-- ============================================================
-- Supabase Auth writes to auth.users on sign-up but does NOT touch public.profiles.
-- Without this trigger every scan insert would fail with a foreign key violation
-- because scans.user_id references public.profiles(id).

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
    insert into public.profiles (id, email)
    values (new.id, new.email);
    return new;
end;
$$;

create trigger on_auth_user_created
    after insert on auth.users
    for each row execute procedure public.handle_new_user();

-- ============================================================
-- 2. Confidence range constraint
-- ============================================================
-- Confidence is a softmax probability, always in [0, 1].

alter table public.scans
    add constraint scans_confidence_range
    check (confidence is null or (confidence >= 0 and confidence <= 1));

-- ============================================================
-- 3. Prediction label constraint
-- ============================================================
-- Only values from the canonical label set (or the Phase-1 binary labels)
-- may be stored. Expand this list in a new migration when Phase 3 adds
-- squamous_cell_carcinoma and seborrheic_keratosis.

alter table public.scans
    add constraint scans_prediction_valid
    check (prediction is null or prediction in (
        -- Phase 1 binary labels
        'suspicious',
        'non_suspicious',
        -- Phase 2 HAM10000 canonical labels
        'melanoma',
        'nevus',
        'basal_cell_carcinoma',
        'actinic_keratosis',
        'benign_keratosis',
        'dermatofibroma',
        'vascular_lesion'
    ));
