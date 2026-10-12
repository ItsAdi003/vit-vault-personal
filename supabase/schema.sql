-- =====================================================================
-- VIT Media Vault: database schema for a NEW Supabase project.
--
-- Reconstructed from how the application code uses the database
-- (client/src and ai-service/). It has NOT been run against a fresh
-- project yet; read the "Verify" notes at the bottom after running it.
--
-- How to use
--   1. Create a free project at supabase.com.
--   2. SQL Editor -> paste this file -> Run.
--   3. Edit is_admin() below (or run the UPDATE at the bottom) so your own
--      e-mail is the admin.
--   4. Authentication -> Providers -> Google: enable it, and add
--      http://localhost:5173/gallery to Authentication -> URL Configuration
--      -> Redirect URLs.
--   5. Copy Project URL, anon key and service_role key into
--      client/.env.local and ai-service/.env.
-- =====================================================================

create extension if not exists vector;      -- pgvector: CLIP image embeddings (512-D)
create extension if not exists pgcrypto;    -- gen_random_uuid()

-- ---------------------------------------------------------------- users
-- The app embeds `users:uploader_id(email)` in media queries, which needs a
-- public table that PostgREST can see (auth.users is not exposed).
create table if not exists public.users (
  id         uuid primary key references auth.users(id) on delete cascade,
  email      text not null,
  created_at timestamptz not null default now()
);

create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into public.users (id, email) values (new.id, new.email)
  on conflict (id) do update set email = excluded.email;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- Back-fill anyone who signed up before the trigger existed.
insert into public.users (id, email)
select id, email from auth.users
on conflict (id) do nothing;

-- --------------------------------------------------------------- events
-- One row per folder ("event"). `type` is the ground-truth label used to
-- train / evaluate the Naive Bayes category classifier.
create table if not exists public.events (
  id            uuid primary key default gen_random_uuid(),
  title         text not null,
  type          text not null check (type in ('INSTITUTIONAL', 'COMMITTEE')),
  event_date    timestamptz not null default now(),
  organizer     text,
  created_by    uuid references public.users(id) on delete set null,
  collaborators text[] not null default '{}',   -- lower-cased e-mails
  created_at    timestamptz not null default now()
);

-- ---------------------------------------------------------------- media
create table if not exists public.media (
  id              uuid primary key default gen_random_uuid(),
  event_id        uuid references public.events(id) on delete cascade,
  uploader_id     uuid references public.users(id) on delete set null,
  file_path       text not null,                -- path inside the vault-media bucket
  status          text not null default 'PENDING'
                  check (status in ('PENDING', 'APPROVED', 'REJECTED')),
  metadata        jsonb not null default '{}'::jsonb,   -- custom_name, description, is_cover, predicted_category, ...
  ai_tags         text[] not null default '{}',         -- CLIP zero-shot tags
  perception_hash text,                                 -- 64-bit pHash as hex
  image_embedding vector(512),                          -- CLIP image embedding
  created_at      timestamptz not null default now()
);

create index if not exists media_event_id_idx on public.media (event_id);
create index if not exists media_status_idx   on public.media (status);
create index if not exists media_embedding_idx
  on public.media using hnsw (image_embedding vector_cosine_ops);

-- ---------------------------------------------------------------- admin
-- EDIT THE E-MAIL BELOW: this account may manage every folder (it mirrors
-- VITE_ADMIN_EMAIL in client/.env.local).
create or replace function public.is_admin()
returns boolean language sql stable as $$
  select lower(coalesce(auth.jwt() ->> 'email', '')) = any (array['raunak.baweja@vit.edu.in']);
$$;

create or replace function public.can_manage_event(ev public.events)
returns boolean language sql stable as $$
  select ev.created_by = auth.uid()
      or lower(coalesce(auth.jwt() ->> 'email', '')) = any (ev.collaborators)
      or public.is_admin();
$$;

-- ------------------------------------------------- row level security
alter table public.users  enable row level security;
alter table public.events enable row level security;
alter table public.media  enable row level security;

drop policy if exists "users: read"        on public.users;
drop policy if exists "events: read"       on public.events;
drop policy if exists "events: insert own" on public.events;
drop policy if exists "events: manage"     on public.events;
drop policy if exists "events: delete"     on public.events;
drop policy if exists "media: read"        on public.media;
drop policy if exists "media: insert own"  on public.media;
drop policy if exists "media: update"      on public.media;
drop policy if exists "media: delete"      on public.media;

create policy "users: read" on public.users
  for select to authenticated using (true);

create policy "events: read" on public.events
  for select to authenticated using (true);
create policy "events: insert own" on public.events
  for insert to authenticated with check (created_by = auth.uid());
create policy "events: manage" on public.events
  for update to authenticated using (public.can_manage_event(events)) with check (public.can_manage_event(events));
create policy "events: delete" on public.events
  for delete to authenticated using (created_by = auth.uid() or public.is_admin());

create policy "media: read" on public.media
  for select to authenticated using (true);
create policy "media: insert own" on public.media
  for insert to authenticated with check (uploader_id = auth.uid());
create policy "media: update" on public.media
  for update to authenticated
  using (
    uploader_id = auth.uid() or public.is_admin()
    or exists (select 1 from public.events e where e.id = media.event_id and public.can_manage_event(e))
  );
create policy "media: delete" on public.media
  for delete to authenticated
  using (
    uploader_id = auth.uid() or public.is_admin()
    or exists (select 1 from public.events e where e.id = media.event_id and e.created_by = auth.uid())
  );

-- ------------------------------------------------------------- storage
insert into storage.buckets (id, name, public)
values ('vault-media', 'vault-media', true)
on conflict (id) do nothing;

drop policy if exists "vault-media: public read"   on storage.objects;
drop policy if exists "vault-media: upload own"    on storage.objects;
drop policy if exists "vault-media: delete own"    on storage.objects;

create policy "vault-media: public read" on storage.objects
  for select using (bucket_id = 'vault-media');
-- the client uploads to "<user id>/<file>"
create policy "vault-media: upload own" on storage.objects
  for insert to authenticated
  with check (bucket_id = 'vault-media' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "vault-media: delete own" on storage.objects
  for delete to authenticated
  using (bucket_id = 'vault-media' and ((storage.foldername(name))[1] = auth.uid()::text or public.is_admin()));

-- =====================================================================
-- Verify (run after the script):
--   select count(*) from public.media;                       -- 0, no error
--   select extname from pg_extension where extname = 'vector';
--   select id, public from storage.buckets;                  -- vault-media | true
-- The AI service uses the service_role key, which bypasses RLS, so it can
-- always read/write `media` regardless of the policies above.
-- =====================================================================
