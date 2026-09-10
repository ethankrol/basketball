begin;
-- First D1 competition years, including transitional participation.
update public.team_memberships set first_season=2002,
 source='https://gomastodons.com/sports/2014/4/23/mens-basketball-history',
 notes='First D1 basketball season 2001-02, before full membership.'
where team_id=1236;
update public.team_memberships set first_season=2005,
 source='https://gobison.com/news/2005/2/23/96164',
 notes='Both Dakota State programs competed as provisional D1 schools in 2004-05.'
where team_id in (1295,1355);
update public.team_memberships set first_season=2001,
 source='https://gwusports.com/news/2011/7/5/7_5_2011_2142.aspx',
 notes='D1 move before 2000-01; legacy starts reflected later data coverage.'
where team_id=1205;
insert into public.team_season_status(team_id,season,competed,reclassifying,postseason_eligible,source)
values
(1295,2005,true,true,false,'https://gobison.com/news/2005/2/23/96164'),
(1355,2005,true,true,false,'https://gobison.com/news/2005/2/23/96164');
commit;
