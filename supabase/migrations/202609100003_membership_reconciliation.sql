begin;
-- Houston left D1 after 1989 and returned for 2007-08; a single min/max
-- interval in the old dataset incorrectly made it D1 throughout the gap.
update public.team_memberships set last_season=1989,
 source='https://hcuhuskies.com/news/2011/8/12/NCAAannounce.aspx',
 notes='First D1 stint; initial 1985 boundary inherited from legacy archive coverage.'
where team_id=1223 and first_season=1985;
insert into public.team_memberships values
(1223,2008,null,'https://hcuhuskies.com/news/2007/10/25/719.aspx','Returned as a transitioning D1 program in 2007-08.');
insert into public.team_season_status(team_id,season,competed,source) values
(1126,2021,false,'https://bcuathletics.com/news/2020/10/27/general-bethune-cookman-to-opt-out-of-all-sports-in-2020-21.aspx'),
(1271,2021,false,'https://umeshawksports.com/news/2020/11/19/general-hawk-athletics-will-not-compete-in-the-spring-of-2020-21-athletic-year.aspx');
update public.team_memberships set first_season=2000,
 source='https://goislanders.com/sports/mens-basketball/schedule/1999-2000?grid=true',
 notes='First basketball season 1999-2000; legacy first-observed season was 2003.'
where team_id=1394;
update public.team_memberships set first_season=2002,
 source='https://bscsports.net/staff.aspx?staff=85',
 notes='D1 competition began in 2001; season-ending convention gives 2002. Last basketball season retained as 2006.'
where team_id=1128;
commit;
