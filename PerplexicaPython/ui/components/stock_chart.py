import streamlit as st

def render_ticker_widget(symbol: str, theme: str = "light"):
    """ Renders the TradingView widget for a stock symbol. """
    if not symbol:
        return
        
    st.markdown(f"### 📈 Stock Chart: {symbol}")
    
    # Create HTML/JS for TradingView
    # Note: Streamlit's st.components.v1.html is needed for embedding JS
    tv_html = f"""
    <div class="tradingview-widget-container" style="height: 400px; width: 100%;">
      <div id="tradingview_chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.mediumWidget({{
        "symbols": [["{symbol}", "{symbol}"]],
        "chartOnly": false,
        "width": "100%",
        "height": 400,
        "locale": "en",
        "colorTheme": "{theme}",
        "gridLineColor": "rgba(240, 243, 250, 0)",
        "fontColor": "#787B86",
        "isTransparent": false,
        "autosize": true,
        "showFloatingTooltip": true,
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    st.components.v1.html(tv_html, height=420)
