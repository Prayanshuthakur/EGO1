import re
from typing import Optional

# Common company name to ticker symbol mappings
COMPANY_TICKER_MAP = {
    'apple': 'NASDAQ:AAPL',
    'microsoft': 'NASDAQ:MSFT',
    'google': 'NASDAQ:GOOGL',
    'alphabet': 'NASDAQ:GOOGL',
    'meta': 'NASDAQ:META',
    'facebook': 'NASDAQ:META',
    'tesla': 'NASDAQ:TSLA',
    'nvidia': 'NASDAQ:NVDA',
    'netflix': 'NASDAQ:NFLX',
    'adobe': 'NASDAQ:ADBE',
    'salesforce': 'NYSE:CRM',
    'oracle': 'NYSE:ORCL',
    'intel': 'NASDAQ:INTC',
    'amd': 'NASDAQ:AMD',
    'ibm': 'NYSE:IBM',
    'cisco': 'NASDAQ:CSCO',
    'uber': 'NYSE:UBER',
    'airbnb': 'NASDAQ:ABNB',
    'spotify': 'NYSE:SPOT',
    'paypal': 'NASDAQ:PYPL',
    'square': 'NYSE:SQ',
    'block': 'NYSE:SQ',
    'twitter': 'NYSE:X',
    'x': 'NYSE:X',
    'snap': 'NYSE:SNAP',
    'snapchat': 'NYSE:SNAP',
    'zoom': 'NASDAQ:ZM'
}

MARKET_KEYWORDS = [
    'stock', 'share', 'price', 'market', 'trading', 'trade', 'invest',
    'ticker', 'chart', 'technical analysis', 'market cap', 'valuation',
    'earnings', 'revenue', 'profit', 'loss', 'p/e', 'dividend',
    'performance', 'quote', '$', 'nasdaq', 'nyse', 'doing', 'up', 'down'
]

def detect_company_ticker(text: str) -> Optional[str]:
    """
    Detects a company ticker symbol from the given text.
    Ported from TypeScript version in lib/company-ticker-map.ts.
    """
    lower_text = text.lower()
    
    # Check if the query is market-related
    is_market_query = any(kw in lower_text for kw in MARKET_KEYWORDS)
    
    market_patterns = [
        r'how\s+is\s+\w+\s+doing',
        r"what('s|\s+is)\s+\w+\s+stock",
        r'\$[A-Z]+'
    ]
    has_market_pattern = any(re.search(pattern, text, re.IGNORECASE) for pattern in market_patterns)
    
    if not is_market_query and not has_market_pattern:
        return None
        
    # Check for direct ticker mentions
    ticker_patterns = [
        r'\$([A-Z]{1,5})\b',
        r'\b([A-Z]{1,5})\s+(?:stock|share|price|chart)',
        r'\b(NYSE|NASDAQ|AMEX):([A-Z.]{1,5})\b'
    ]
    
    for pattern in ticker_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if 'NYSE|NASDAQ' in pattern:
                return match.group(0).upper()
            elif match.lastindex >= 1:
                ticker = match.group(1).upper()
                # Check if it's in our map values
                for t in COMPANY_TICKER_MAP.values():
                    if ticker in t:
                        return t
                        
    # Check for company name + market keyword
    sorted_companies = sorted(COMPANY_TICKER_MAP.keys(), key=len, reverse=True)
    for company in sorted_companies:
        if re.search(r'\b' + re.escape(company) + r'\b', lower_text):
            return COMPANY_TICKER_MAP[company]
            
    return None
