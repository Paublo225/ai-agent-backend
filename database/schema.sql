-- Supabase/PostgreSQL schema for the appliance repair agent
create table if not exists documents (
    document_id text primary key,
    filename text not null,
    appliance_type text,
    brand text,
    upload_date timestamptz default now(),
    total_pages integer default 0,
    metadata jsonb default '{}'::jsonb
);

create table if not exists conversations (
    id uuid primary key default gen_random_uuid(),
    user_id uuid,
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create table if not exists messages (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid references conversations(id) on delete cascade,
    role text not null,
    content text not null,
    citations jsonb default '[]'::jsonb,
    created_at timestamptz default now()
);

create table if not exists user_sessions (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid references conversations(id) on delete cascade,
    status text default 'active',
    metadata jsonb default '{}'::jsonb,
    created_at timestamptz default now()
);

create table if not exists uploads (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid references conversations(id) on delete set null,
    file_path text not null,
    mime_type text,
    created_at timestamptz default now()
);

create index if not exists idx_messages_conversation_id on messages(conversation_id);
create index if not exists idx_documents_brand on documents(brand);
create index if not exists idx_documents_appliance_type on documents(appliance_type);
