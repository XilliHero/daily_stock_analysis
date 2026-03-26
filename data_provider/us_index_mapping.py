# -*- coding: utf-8 -*-
"""
===================================
美股指数与股票代码工具
===================================

提供：
1. 美股指数代码映射（如 SPX -> ^GSPC）
2. 美股股票代码识别（AAPL、TSLA 等）
3. 加拿大股票代码识别（MDA.TO 等 TSX 代码）
4. 加密货币代码识别（BTC-USD 等）
5. 统一 yfinance 支持检测

美股指数在 Yahoo Finance 中需使用 ^ 前缀，与股票代码不同。
"""

import re

# 美股代码正则：1-5 个大写字母，可选 .X 后缀（如 BRK.B）
_US_STOCK_PATTERN = re.compile(r'^[A-Z]{1,5}(\.[A-Z])?$')

# 加拿大股票代码正则：1-5 个大写字母 + .TO / .TSX / .V 后缀（TSX 上市股票）
_CA_STOCK_PATTERN = re.compile(r'^[A-Z]{1,5}\.(TO|TSX|V)$')

# 加密货币代码正则：如 BTC-USD, ETH-USD, ETH-BTC
_CRYPTO_PATTERN = re.compile(r'^[A-Z]{2,6}-[A-Z]{2,5}$')


# 用户输入 -> (Yahoo Finance 符号, 中文名称)
US_INDEX_MAPPING = {
    # 标普 500
    'SPX': ('^GSPC', '标普500指数'),
    '^GSPC': ('^GSPC', '标普500指数'),
    'GSPC': ('^GSPC', '标普500指数'),
    # 道琼斯工业平均指数
    'DJI': ('^DJI', '道琼斯工业指数'),
    '^DJI': ('^DJI', '道琼斯工业指数'),
    'DJIA': ('^DJI', '道琼斯工业指数'),
    # 纳斯达克综合指数
    'IXIC': ('^IXIC', '纳斯达克综合指数'),
    '^IXIC': ('^IXIC', '纳斯达克综合指数'),
    'NASDAQ': ('^IXIC', '纳斯达克综合指数'),
    # 纳斯达克 100
    'NDX': ('^NDX', '纳斯达克100指数'),
    '^NDX': ('^NDX', '纳斯达克100指数'),
    # VIX 波动率指数
    'VIX': ('^VIX', 'VIX恐慌指数'),
    '^VIX': ('^VIX', 'VIX恐慌指数'),
    # 罗素 2000
    'RUT': ('^RUT', '罗素2000指数'),
    '^RUT': ('^RUT', '罗素2000指数'),
}


def is_us_index_code(code: str) -> bool:
    """
    判断代码是否为美股指数符号。

    Args:
        code: 股票/指数代码，如 'SPX', 'DJI'

    Returns:
        True 表示是已知美股指数符号，否则 False

    Examples:
        >>> is_us_index_code('SPX')
        True
        >>> is_us_index_code('AAPL')
        False
    """
    return (code or '').strip().upper() in US_INDEX_MAPPING


def is_us_stock_code(code: str) -> bool:
    """
    判断代码是否为美股股票符号（排除美股指数）。

    美股股票代码为 1-5 个大写字母，可选 .X 后缀如 BRK.B。
    美股指数（SPX、DJI 等）明确排除。

    Args:
        code: 股票代码，如 'AAPL', 'TSLA', 'BRK.B'

    Returns:
        True 表示是美股股票符号，否则 False

    Examples:
        >>> is_us_stock_code('AAPL')
        True
        >>> is_us_stock_code('TSLA')
        True
        >>> is_us_stock_code('BRK.B')
        True
        >>> is_us_stock_code('SPX')
        False
        >>> is_us_stock_code('600519')
        False
    """
    normalized = (code or '').strip().upper()
    # 美股指数不是股票
    if normalized in US_INDEX_MAPPING:
        return False
    return bool(_US_STOCK_PATTERN.match(normalized))


def get_us_index_yf_symbol(code: str) -> tuple:
    """
    获取美股指数的 Yahoo Finance 符号与中文名称。

    Args:
        code: 用户输入，如 'SPX', '^GSPC', 'DJI'

    Returns:
        (yf_symbol, chinese_name) 元组，未找到时返回 (None, None)。

    Examples:
        >>> get_us_index_yf_symbol('SPX')
        ('^GSPC', '标普500指数')
        >>> get_us_index_yf_symbol('AAPL')
        (None, None)
    """
    normalized = (code or '').strip().upper()
    return US_INDEX_MAPPING.get(normalized, (None, None))


def is_ca_stock_code(code: str) -> bool:
    """
    判断代码是否为加拿大股票（TSX 上市），如 MDA.TO。

    支持后缀：.TO (TSX 主板), .TSX (别名), .V (TSX Venture)

    Args:
        code: 股票代码，如 'MDA.TO', 'SU.TO'

    Returns:
        True 表示是加拿大 TSX 股票代码，否则 False

    Examples:
        >>> is_ca_stock_code('MDA.TO')
        True
        >>> is_ca_stock_code('AAPL')
        False
    """
    normalized = (code or '').strip().upper()
    return bool(_CA_STOCK_PATTERN.match(normalized))


def is_crypto_code(code: str) -> bool:
    """
    判断代码是否为加密货币代码，如 BTC-USD, ETH-USD。

    格式：2-6 个大写字母 + 连字符 + 2-5 个大写字母（计价货币）。

    Args:
        code: 代码，如 'BTC-USD', 'ETH-BTC'

    Returns:
        True 表示是加密货币代码，否则 False

    Examples:
        >>> is_crypto_code('BTC-USD')
        True
        >>> is_crypto_code('AAPL')
        False
    """
    normalized = (code or '').strip().upper()
    return bool(_CRYPTO_PATTERN.match(normalized))


def is_yfinance_supported(code: str) -> bool:
    """
    判断代码是否可以由 yfinance 处理（美股、加拿大股、加密货币、美股指数）。

    用于将请求快速路由到 YfinanceFetcher，而不是先尝试只支持 A 股的数据源。

    Args:
        code: 股票/加密货币代码

    Returns:
        True 表示 yfinance 支持该代码，否则 False

    Examples:
        >>> is_yfinance_supported('TSLA')
        True
        >>> is_yfinance_supported('BTC-USD')
        True
        >>> is_yfinance_supported('MDA.TO')
        True
        >>> is_yfinance_supported('SPX')
        True
        >>> is_yfinance_supported('600519')
        False
    """
    normalized = (code or '').strip().upper()
    return (
        is_us_index_code(normalized)
        or is_us_stock_code(normalized)
        or is_ca_stock_code(normalized)
        or is_crypto_code(normalized)
    )
