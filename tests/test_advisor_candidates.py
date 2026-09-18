from src.advisor.candidates import Candidate, fetch_candidates


class _Stock:
    def __init__(self, ticker, name, strategy_count, sector="Unknown"):
        self.ticker = ticker
        self.name = name
        self.strategy_count = strategy_count
        self.sector = sector


class _Board:
    def __init__(self, stocks):
        self.stocks = stocks


def test_merges_and_ranks_by_conviction():
    board = _Board([_Stock("CNQ.TO", "Cdn Natural", 3, "Energy"),
                    _Stock("INGR", "Ingredion", 4, "Consumer")])
    cands = fetch_candidates(limit=10, board_provider=lambda: board,
                             watchlist_provider=lambda: ["OKLO"])
    tickers = [c.ticker for c in cands]
    assert tickers[0] == "INGR"          # 4/4 ranks above 3/4
    assert "CNQ.TO" in tickers
    assert "OKLO" in tickers             # watchlist adds names not on the board


def test_dedup_and_limit():
    board = _Board([_Stock("INGR", "Ingredion", 4)])
    cands = fetch_candidates(limit=1, board_provider=lambda: board,
                             watchlist_provider=lambda: ["INGR", "OKLO"])
    assert len(cands) == 1
    assert cands[0].ticker == "INGR"     # dedup: INGR not repeated; limit respected


def test_candidate_carries_verdict_when_present():
    board = _Board([_Stock("INGR", "Ingredion", 4)])
    verdicts = {"INGR": {"verdict": "Buy", "sentiment": "Bullish"}}
    cands = fetch_candidates(limit=5, board_provider=lambda: board,
                             watchlist_provider=lambda: [], verdict_provider=lambda t: verdicts.get(t))
    assert cands[0].verdict == "Buy"
