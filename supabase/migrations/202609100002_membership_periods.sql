begin;
create extension if not exists btree_gist with schema extensions;
create table public.team_memberships (
    team_id bigint not null references public.teams(team_id),
    first_season integer not null,
    last_season integer,
    source text not null,
    notes text,
    primary key (team_id, first_season),
    check (last_season is null or last_season >= first_season),
    exclude using gist (team_id with =, int4range(first_season, last_season, '[]') with &&)
);
-- 2025 was the legacy dataset coverage boundary, not a departure date.
-- Earlier endpoints are retained as historical assertions with explicit provenance.
insert into public.team_memberships(team_id, first_season, last_season, source, notes)
select team_id, first_d1_season, case when last_d1_season=2025 then null else last_d1_season end,
       'legacy teams metadata imported 2026-09-10',
       'Historical starts/endpoints inherited; transitioning seasons may need source reconciliation.'
from public.teams;

-- Local stable ID follows the existing catalog; don't renumber existing schools.
insert into public.teams(team_id, team_name, first_d1_season, last_d1_season)
values (1481, 'New Haven', 2026, 2026);
insert into public.team_spellings(team_spelling, team_id) values ('new haven',1481);
insert into public.team_memberships values
(1481,2026,null,'https://www.newhaven.edu/news/releases/2025/northeast-conference-invitation.php',
 'Competes in D1 from 2025-26 while reclassifying; postseason eligibility is separate.');
update public.team_memberships set last_season=2026,
 source='https://www.francis.edu/D3TransitionFAQ',
 notes='Remains D1 in 2025-26; Division III competition starts fall 2026 (season 2027).'
where team_id=(select team_id from public.teams where team_name='St Francis PA');

create table public.team_season_status (
    team_id bigint not null references public.teams(team_id),
    season integer not null,
    competed boolean not null default true,
    reclassifying boolean,
    postseason_eligible boolean,
    source text not null,
    primary key (team_id, season)
);
insert into public.team_season_status(team_id,season,competed,source)
select team_id,2021,false,
'https://ivyleague.com/news/2020/11/12/general-ivy-league-outlines-intercollegiate-athletics-plans-no-competition-for-winter-sports.aspx'
from public.teams where team_name in ('Brown','Columbia','Cornell','Dartmouth','Harvard','Penn','Princeton','Yale');
insert into public.team_season_status values
(1481,2026,true,true,false,'https://www.newhaven.edu/news/releases/2025/northeast-conference-invitation.php');

create table public.team_elo_seasons (
    state_hash text not null,
    season integer not null,
    team_id bigint not null references public.teams(team_id),
    ending_elo double precision not null,
    initial_season integer not null,
    prior_state_hash text,
    elo_version text not null,
    carryover double precision not null check(carryover between 0 and 1),
    source_hash text not null,
    uploaded_at timestamptz not null default now(),
    primary key (state_hash, team_id)
);
alter table public.team_memberships enable row level security;
alter table public.team_season_status enable row level security;
alter table public.team_elo_seasons enable row level security;
revoke all on public.team_memberships,public.team_season_status,public.team_elo_seasons from anon,authenticated;
grant select,insert,update on public.team_memberships,public.team_season_status,public.team_elo_seasons to service_role;

-- Same typed feature columns; v4 changes membership selection/provenance.
do $$ begin
 execute replace(pg_get_functiondef('public.upload_feature_backfill(jsonb,jsonb,jsonb,text)'::regprocedure),
                 'is distinct from ''d1-v3''', 'is distinct from ''d1-v4''');
end $$;
notify pgrst,'reload schema';
commit;
