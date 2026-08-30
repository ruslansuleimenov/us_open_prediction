import pytest

from usopen.data import ROUND_ORDER, SENTINEL_COLS, load_matches, SERIES_CANONICAL

BRISBANE = (
    "Brisbane,2025-01-01,ATP250,Outdoor,Hard,1st Round,3,"
    "Ivanov A.,Petrov B.,Ivanov A.,-1,50,-1,1200,-1.0,2.5,6-3 6-4"
)
CITI_OPEN = (
    "Citi Open,2025-07-29,ATP500,Outdoor,Hard,1st Round,3,"
    "Karatsev A.,Garin C.,Garin C.,102,115,605,518,1.57,2.38,4-6 2-6"
)


class TestLoadMatches:
    def test_sentinels_become_nan(self, write_csv):
        df = load_matches(write_csv(BRISBANE, CITI_OPEN))
        brisbane = df.loc[df["Tournament"] == "Brisbane"].iloc[0]
        citi = df.loc[df["Tournament"] == "Citi Open"].iloc[0]

        # no -1 survived anywhere
        assert not (df[SENTINEL_COLS] == -1).any().any()

        # where it was, there is now NaN
        assert brisbane[["Rank_1", "Pts_1", "Odd_1"]].isna().all()

        # where it was not, values are untouched
        assert brisbane["Rank_2"] == 50
        assert citi[SENTINEL_COLS].notna().all()
        assert citi["Rank_1"] == 102

    def test_sentinel_columns_stay_numeric(self, write_csv):
        df = load_matches(write_csv(BRISBANE, CITI_OPEN))

        # the bug this guards against: pd.NA turned these columns into object
        assert (df[SENTINEL_COLS].dtypes != "object").all()
        assert df["Rank_1"].dtype == "float64"

    def test_unknown_round_is_rejected(self, write_csv):
        unknown = BRISBANE.replace("1st Round", "Qualifying")
        assert "Qualifying" not in ROUND_ORDER

        # a mix of a valid and an unknown row: on a single-row fixture the
        # loader's .all() assert would be indistinguishable from .any()
        with pytest.raises(AssertionError):
            load_matches(write_csv(CITI_OPEN, unknown))

    def test_unknown_series_is_rejected(self, write_csv):
        unknown = BRISBANE.replace("ATP250", "ATP333")
        assert "ATP333" not in SERIES_CANONICAL
        with pytest.raises(AssertionError):
            load_matches(write_csv(CITI_OPEN, unknown))

    def test_sorted_by_date_then_round(self, write_csv):
        # the file is deliberately out of order: the later date comes first,
        # and within one day the semifinal precedes the quarterfinal
        late = CITI_OPEN
        semi = BRISBANE.replace("1st Round", "Semifinals")
        quarter = BRISBANE.replace("1st Round", "Quarterfinals")

        df = load_matches(write_csv(late, semi, quarter))

        assert df["Date"].is_monotonic_increasing
        assert list(df["Round"]) == ["Quarterfinals", "Semifinals", "1st Round"]
        assert list(df["round_order"]) == [5, 6, 1]

    def test_index_is_reset(self, write_csv):
        df = load_matches(write_csv(CITI_OPEN, BRISBANE))

        assert list(df.index) == [0, 1]

    def test_round_robin_precedes_knockout(self, write_csv):
        # the Tour Finals group stage comes before the knockout rounds,
        # even though "Round Robin" sorts nowhere near them lexically
        rr = BRISBANE.replace("ATP250,Outdoor,Hard,1st Round", "Masters Cup,Indoor,Hard,Round Robin")
        final = BRISBANE.replace("1st Round", "The Final")
        semi = BRISBANE.replace("1st Round", "Semifinals")

        df = load_matches(write_csv(final, semi, rr))

        assert list(df["Round"]) == ["Round Robin", "Semifinals", "The Final"]
