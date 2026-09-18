from src.advisor.classify import classify_asset, sector_of


def test_classify_asset():
    assert classify_asset("AAPL") == "equity"
    assert classify_asset("cnq.to") == "equity"       # Canadian equity, case-insensitive
    assert classify_asset("BTC-USD") == "crypto"
    assert classify_asset("BND") == "fixed_income"
    assert classify_asset("CORN") == "commodity"
    assert classify_asset("CASH") == "cash"


def test_sector_lookup_with_default():
    lookup = {"CNQ.TO": "Energy", "AAPL": "Technology"}
    assert sector_of("CNQ.TO", lookup) == "Energy"
    assert sector_of("AAPL", lookup) == "Technology"
    assert sector_of("ZZZZ", lookup) == "Unknown"     # unknown -> Unknown
    assert sector_of("AAPL", None) == "Unknown"        # no lookup -> Unknown
