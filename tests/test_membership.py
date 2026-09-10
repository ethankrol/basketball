import unittest

from etl.features import canonical_games
from etl.membership import active_teams, season_nonparticipants


class MembershipTests(unittest.TestCase):
    teams = [{'team_id':1,'team_name':'Returning school'}, {'team_id':2,'team_name':'New school'}]

    def test_active_members_do_not_expire_at_archive_boundary(self):
        periods = [{'team_id':1,'first_season':2001,'last_season':None}]
        self.assertEqual(active_teams(self.teams,periods,2027), {1:'Returning school'})

    def test_departure_entry_and_return_boundaries(self):
        periods = [{'team_id':1,'first_season':2001,'last_season':2005},
                   {'team_id':1,'first_season':2010,'last_season':None},
                   {'team_id':2,'first_season':2006,'last_season':None}]
        self.assertEqual(set(active_teams(self.teams,periods,2005)),{1})
        self.assertEqual(set(active_teams(self.teams,periods,2006)),{2})
        self.assertEqual(set(active_teams(self.teams,periods,2010)),{1,2})

    def test_overlap_is_rejected_even_outside_requested_year(self):
        periods = [{'team_id':1,'first_season':2001,'last_season':2005},
                   {'team_id':1,'first_season':2005,'last_season':None}]
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            active_teams(self.teams,periods,2026)

    def test_membership_is_not_inferred_from_missing_metadata(self):
        with self.assertRaisesRegex(ValueError,'Missing team_memberships'):
            active_teams(self.teams,None,2026)

    def test_canceled_season_does_not_end_membership(self):
        periods = [{'team_id':1,'first_season':2001,'last_season':None}]
        status = {'team_season_status':[{'team_id':1,'season':2021,'competed':False}]}
        members = active_teams(self.teams,periods,2021)
        nonparticipants = season_nonparticipants(status,2021)
        games, _ = canonical_games([],{},members,2021,nonparticipants)
        self.assertEqual(games,[])
        self.assertEqual(active_teams(self.teams,periods,2022),members)
        with self.assertRaisesRegex(ValueError,'without mapped'):
            canonical_games([],{},members,2022,season_nonparticipants(status,2022))
