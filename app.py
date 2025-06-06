import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from io import StringIO
from pathlib import Path
from backend.stock_valuation import create_stock_valuation
from backend.technical_analysis import analyze_stock

st.set_page_config(
    page_title="Stock Valuation Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "show_advanced" not in st.session_state:
    st.session_state.show_advanced = False

st.title("📈 NiveshLakshya: Stock Valuation & Technical Analysis Tool")
st.markdown("""
This tool helps you estimate the intrinsic value of stocks using multiple valuation methods and technical analysis.
Enter the financial metrics and click 'Run Valuation' to analyze.
""")
st.markdown("""
**Disclaimer**: This tool is for educational purposes only. The valuations provided should not be considered as 
investment advice or recommendations. Always consult your financial advisor before making investment decisions.
""")

tab1, tab2 = st.tabs(["Valuation Analysis", "Help"])

with tab1:
    
    input_col, results_col = st.columns([1, 2])
    
    with input_col:
        st.subheader("Stock Information")
        company_name = st.text_input("Company Name", value="TCS")
        
        ticker_col1, ticker_col2 = st.columns([2, 1])
        with ticker_col1:
            ticker_symbol = st.text_input("Stock Ticker Symbol", value="TCS")
        with ticker_col2:
            exchange = st.selectbox("Exchange", ["NSE (India)", "NYSE (US)", "NASDAQ (US)"])
        
        current_price = st.number_input("Current Market Price (₹)", value=3800.0, min_value=0.01)
        industry = st.selectbox("Industry", [
            "Information Technology", "Banking", "FMCG", "Pharmaceuticals",
            "Automobile", "Energy", "Telecommunications", "Manufacturing", "Other"
        ])
        profile = st.selectbox("Investor Profile", ["Neutral", "Optimistic", "Pessimistic"])
        
        with st.expander("Earnings & Valuation Metrics", expanded=True):
            eps = st.number_input("Earnings Per Share (₹)", value=150.0, min_value=0.0)
            book_value = st.number_input("Book Value Per Share (₹)", value=700.0, min_value=0.0)
            industry_pe = st.number_input("Industry P/E Ratio", value=25.0, min_value=0.0, step=0.5)
            eps_growth = st.slider("EPS Growth Rate (%)", 0.0, 100.0, 12.0, 0.5)
            profit_margin = st.slider("Profit Margin (%)", 0.0, 100.0, 18.0, 0.5)
        
        with st.expander("Cash Flow & Dividend Metrics"):
            dividend = st.number_input("Dividend Per Share (₹)", value=45.0, min_value=0.0)
            fcf = st.number_input("Free Cash Flow Per Share (₹)", value=120.0, min_value=0.0)
        
        with st.expander("Financial Health Metrics"):
            roe = st.slider("Return on Equity (%)", 0.0, 100.0, 23.0, 0.5)
            de_ratio = st.number_input("Debt to Equity Ratio", value=0.3, min_value=0.0, step=0.1)
            current_ratio = st.number_input("Current Ratio", value=2.1, min_value=0.01, step=0.1)
            
        with st.expander("Advanced Settings"):
            st.session_state.show_advanced = True
            
            dcf_required_return = st.slider("DCF Required Return (%)", 8.0, 25.0, 15.0, 0.5)
            dcf_terminal_growth = st.slider("DCF Terminal Growth (%)", 2.0, 8.0, 4.0, 0.5)
        
        run_button = st.button("🔍 Run Valuation", type="primary", use_container_width=True)
    
    with results_col:
        if not run_button:
            st.info("👈 Enter your stock data on the left and click 'Run Valuation' to analyze.")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            methods = ['PE', 'PB', 'DCF', 'DDM', 'Graham']
            sample_values = [4200, 3900, 4500, 4100, 3800]
            bars = ax.bar(methods, sample_values, color='lightblue', alpha=0.7)
            ax.axhline(y=3800, color='r', linestyle='--', linewidth=2, label='Current Price ₹3,800')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 50, f'₹{height:,}', 
                     ha='center', va='bottom', alpha=0.7)
                
            ax.set_title('Sample Valuation Chart (Demo)', alpha=0.7)
            ax.set_ylabel('Stock Price (₹)')
            plt.legend()
            st.pyplot(fig)
            
            st.subheader("About Financial Health Indicators")
            help_cols = st.columns(3)
            
            with help_cols[0]:
                st.metric("Return on Equity", "15-25%")
                st.caption("🟢 Excellent: >20%\n🟡 Good: 12-20%\n🔴 Concerning: <12%")
                
            with help_cols[1]:
                st.metric("Debt to Equity", "<1.0")
                st.caption("🟢 Low Debt: <0.5\n🟡 Moderate: 0.5-1.0\n🔴 High Debt: >1.0")
                
            with help_cols[2]:
                st.metric("Current Ratio", ">1.5")
                st.caption("🟢 Strong: >1.5\n🟡 Adequate: 1.0-1.5\n🔴 Concern: <1.0")
    
    if run_button:
        with st.spinner("Analyzing valuation..."):
            try:
                if not company_name.strip():
                    input_col.error("Company name cannot be empty.")
                elif current_price <= 0:
                    input_col.error("Current price must be greater than zero.")
                else:
                    model = create_stock_valuation(company_name, current_price)
                    
                    if model is None:
                        input_col.error("Failed to create valuation model. Please check your inputs.")
                    else:
                        data_set_success = model.set_financial_data(
                            eps=eps,
                            book_value=book_value,
                            dividend=dividend,
                            free_cash_flow=fcf,
                            eps_growth_rate=eps_growth / 100,
                            roe=roe / 100,
                            debt_to_equity=de_ratio,
                            profit_margin=profit_margin / 100,
                            current_ratio=current_ratio,
                            pe_ratio_industry=industry_pe,
                            industry=industry
                        )
                        
                        if not data_set_success:
                            input_col.error("Failed to set financial data. Please check your inputs.")
                        else:
                            if st.session_state.show_advanced:
                                model.pe_valuation()
                                model.pb_valuation()
                                model.graham_number()
                                
                                model.dividend_discount_model(required_return=dcf_required_return/100, 
                                                            terminal_growth=dcf_terminal_growth/100)
                                model.dcf_valuation(required_return=dcf_required_return/100, 
                                                   terminal_growth=dcf_terminal_growth/100)
                                
                                result = model.intrinsic_value_calculator(profile=profile.lower())
                            else:
                                result = model.intrinsic_value_calculator(profile=profile.lower())
                            
                            if "error" in result:
                                input_col.error(f"Calculation error: {result['error']}")
                            else:
                                with results_col:
                                    st.subheader("📊 Valuation Summary")
                                    
                                    metric_cols = st.columns(3)
                                    with metric_cols[0]:
                                        st.metric("Current Price", f"₹{current_price:,.2f}")
                                    with metric_cols[1]:
                                        st.metric("Intrinsic Value", f"₹{result['intrinsic_value']:,.2f}")
                                    with metric_cols[2]:
                                        delta_color = "normal" if result['potential'] >= 0 else "inverse"
                                        
                                        st.metric("Potential", f"{result['potential']}%", 
                                                  delta=f"{result['potential']}%", 
                                                  delta_color=delta_color)
                                    
                                    recommendation = model.get_recommendation(result)
                                    st.info(f"**Recommendation:** {recommendation}", icon="📝")
                                    
                                    st.subheader("Valuation Methods Breakdown")
                                    
                                    valuations_data = []
                                    for method, value in result['individual_valuations'].items():
                                        if method in model.valuation_results:
                                            diff = model.valuation_results[method]["difference"]
                                            weight = result["weights"].get(method, 0)
                                            valuations_data.append({
                                                "Method": method,
                                                "Value (₹)": f"₹{value:,.2f}",
                                                "Difference (%)": f"{diff:.2f}%",
                                                "Weight (%)": f"{weight}%"
                                            })
                                    
                                    if valuations_data:
                                        df_valuations = pd.DataFrame(valuations_data)
                                        st.dataframe(df_valuations, hide_index=True, use_container_width=True)
                                        
                                        csv_data = pd.DataFrame({
                                            "Method": [item["Method"] for item in valuations_data],
                                            "Value (₹)": [item["Value (₹)"].replace("₹", "").replace(",", "") for item in valuations_data],
                                            "Difference (%)": [item["Difference (%)"].replace("%", "") for item in valuations_data],
                                            "Weight (%)": [item["Weight (%)"].replace("%", "") for item in valuations_data]
                                        })
                                        
                                        summary_data = pd.DataFrame({
                                            "Metric": ["Company", "Industry", "Current Price", "Intrinsic Value", "Potential", "Recommendation"],
                                            "Value": [
                                                company_name, 
                                                industry,
                                                current_price,
                                                result["intrinsic_value"],
                                                f"{result['potential']}%",
                                                recommendation
                                            ]
                                        })
                                        
                                        metrics_data = pd.DataFrame({
                                            "Metric": ["EPS", "Book Value", "Dividend", "Free Cash Flow", "EPS Growth", "ROE", 
                                                      "Debt to Equity", "Profit Margin", "Current Ratio", "Industry PE"],
                                            "Value": [
                                                eps, book_value, dividend, fcf, f"{eps_growth}%", f"{roe}%",
                                                de_ratio, f"{profit_margin}%", current_ratio, industry_pe
                                            ]
                                        })
                                        
                                        st.download_button(
                                            label="📊 Export Valuation Data to CSV",
                                            data=pd.concat([summary_data, csv_data, metrics_data]).to_csv(index=False).encode('utf-8'),
                                            file_name=f"{company_name}_valuation_{datetime.now().strftime('%Y%m%d')}.csv",
                                            mime="text/csv"
                                        )
                                    
                                    st.subheader("📉 Valuation Chart")
                                    model.plot_valuation_comparison()
                                    
                                    st.subheader("Financial Health Indicators")
                                    health_cols = st.columns(3)
                                    
                                    with health_cols[0]:
                                        roe_percentage = roe
                                        roe_status = "🟢 Excellent" if roe_percentage > 20 else "🟡 Good" if roe_percentage > 12 else "🔴 Concerning"
                                        st.metric("Return on Equity", f"{roe_percentage:.1f}%")
                                        st.caption(f"Status: {roe_status}")
                                    
                                    with health_cols[1]:
                                        de_status = "🟢 Low Debt" if de_ratio < 0.5 else "🟡 Moderate Debt" if de_ratio < 1 else "🔴 High Debt"
                                        st.metric("Debt to Equity", f"{de_ratio:.2f}")
                                        st.caption(f"Status: {de_status}")
                                    
                                    with health_cols[2]:
                                        cr_status = "🟢 Strong Liquidity" if current_ratio > 1.5 else "🟡 Adequate Liquidity" if current_ratio > 1 else "🔴 Liquidity Concern"
                                        st.metric("Current Ratio", f"{current_ratio:.2f}")
                                        st.caption(f"Status: {cr_status}")
                                
                                    with st.expander("📋 View Detailed Valuation Report"):
                                        report = model.generate_report()
                                        st.text_area("Full Report", report, height=400)
                                        
                                        report_io = StringIO()
                                        report_io.write(report)
                                        
                                        st.download_button(
                                            label="💾 Download Report",
                                            data=report_io.getvalue(),
                                            file_name=f"{company_name}_valuation_{datetime.now().strftime('%Y%m%d')}.txt",
                                            mime="text/plain"
                                        )
                                    
                                    st.subheader("📈 Technical Analysis")
                                    
                                    full_ticker = ticker_symbol
                                    if exchange == "NSE (India)":
                                        full_ticker = f"{ticker_symbol}.NS"
                                    elif exchange == "NYSE (US)":
                                        full_ticker = f"{ticker_symbol}"
                                    elif exchange == "NASDAQ (US)":
                                        full_ticker = f"{ticker_symbol}"
                                    
                                    st.info(f"Analyzing technical indicators for {full_ticker} with fair value of ₹{result['intrinsic_value']:,.2f}")
                                    
                                    try:
                                        with st.spinner("Generating technical analysis..."):
                                            from backend.technical_analysis import analyze_stock
                                            
                                            tech_result, fig, recommendation_text = analyze_stock(
                                                ticker=full_ticker,
                                                fair_value=result['intrinsic_value']
                                            )
                                            
                                            st.pyplot(fig)
                                            
                                            with st.expander("📊 Technical Analysis Recommendation", expanded=True):
                                                st.markdown(f"**RECOMMENDATION:** {tech_result.entry_strategy}")
                                                st.markdown(f"**Confidence Score:** {tech_result.confidence_score}/100")
                                                st.markdown("**REASONING:**")
                                                st.markdown(tech_result.reasoning)
                                                
                                                # Create tabs for Buy and Sell zones
                                                buy_tab, sell_tab = st.tabs(["Buy Zones", "Sell Zones"])
                                                
                                                with buy_tab:
                                                    # Buy zones in a table format
                                                    buy_data = []
                                                    for confidence in ["High", "Medium", "Low"]:
                                                        zones = [z for z in tech_result.buying_zones if z.confidence == confidence]
                                                        for z in sorted(zones, key=lambda x: x.lower):
                                                            buy_data.append({
                                                                "Confidence": confidence,
                                                                "Price Range": f"₹{z.lower:.2f} - ₹{z.upper:.2f}",
                                                                "Description": z.description
                                                            })
                                                    
                                                    if buy_data:
                                                        buy_df = pd.DataFrame(buy_data)
                                                        # Style the dataframe for better visual appeal
                                                        def highlight_confidence(val):
                                                            if val == "High":
                                                                return 'background-color: #004d00; color: white'
                                                            elif val == "Medium":
                                                                return 'background-color: #00b300; color: white'
                                                            elif val == "Low":
                                                                return 'background-color: #99ff99; color: black'
                                                            return ''
                                                        
                                                        styled_buy_df = buy_df.style.map(highlight_confidence, subset=['Confidence'])
                                                        st.dataframe(styled_buy_df, hide_index=True, use_container_width=True)
                                                    else:
                                                        st.write("No buy zones identified")
                                                
                                                with sell_tab:
                                                    # Sell zones in a table format
                                                    sell_data = []
                                                    for confidence in ["High", "Medium", "Low"]:
                                                        zones = [z for z in tech_result.selling_zones if z.confidence == confidence]
                                                        for z in sorted(zones, key=lambda x: x.lower):
                                                            sell_data.append({
                                                                "Confidence": confidence,
                                                                "Price Range": f"₹{z.lower:.2f} - ₹{z.upper:.2f}",
                                                                "Description": z.description
                                                            })
                                                    
                                                    if sell_data:
                                                        sell_df = pd.DataFrame(sell_data)
                                                        # Style the dataframe for better visual appeal
                                                        def highlight_confidence(val):
                                                            if val == "High":
                                                                return 'background-color: #990000; color: white'
                                                            elif val == "Medium":
                                                                return 'background-color: #ff3333; color: white'
                                                            elif val == "Low":
                                                                return 'background-color: #ffb3b3; color: black'
                                                            return ''
                                                        
                                                        styled_sell_df = sell_df.style.map(highlight_confidence, subset=['Confidence'])
                                                        st.dataframe(styled_sell_df, hide_index=True, use_container_width=True)
                                                    else:
                                                        st.write("No sell zones identified")
                                            
                                    except Exception as e:
                                        st.error(f"Error running technical analysis: {str(e)}")
                                        st.error("Please check if the ticker symbol is correct or try again later.")
                
            except ValueError as ve:
                st.error(f"Invalid input: {ve}")
            except Exception as e:
                st.error("An error occurred during valuation calculation.")
                st.error(f"Error details: {str(e)}")
                
                if os.environ.get("STREAMLIT_ENV") == "development":
                    st.exception(e)

with tab2:
    st.subheader("Understanding Valuation Methods")
    
    method_expanders = [
        ("PE Valuation", "Price-to-Earnings valuation compares the stock's price to its earnings per share. A lower P/E ratio relative to peers may indicate an undervalued stock."),
        ("PB Valuation", "Price-to-Book valuation compares the stock's price to its book value per share. This is especially useful for financial companies."),
        ("DCF Valuation", "Discounted Cash Flow analysis estimates the value of a stock based on projected future cash flows, discounted to present value."),
        ("DDM Valuation", "Dividend Discount Model values a stock based on expected future dividends, discounted to present value."),
        ("Graham Number", "A formula developed by Benjamin Graham that calculates a fair value based on earnings per share and book value.")
    ]
    
    for title, description in method_expanders:
        with st.expander(title):
            st.write(description)
    
    st.subheader("Financial Metrics Explained")
    metrics_md = """
    - **EPS (Earnings Per Share)**: Company's profit divided by outstanding shares
    - **Book Value**: Assets minus liabilities per share
    - **ROE (Return on Equity)**: Measures how efficiently a company uses shareholders' equity
    - **Debt to Equity Ratio**: Indicates the company's financial leverage
    - **Current Ratio**: Measures the company's ability to pay short-term obligations
    - **Profit Margin**: The percentage of revenue that translates into profit
    - **Free Cash Flow**: Operating cash flow minus capital expenditures
    """
    st.markdown(metrics_md)
    
    st.subheader("About Technical Analysis")
    st.markdown("""
    The technical analysis section identifies key buy and sell zones based on:
    
    - **Moving Averages**: 50-day, 100-day, and 200-day
    - **Support Levels**: Historical price clusters where buyers entered
    - **Resistance Levels**: Historical price levels where selling pressure increased
    - **Momentum Indicators**: RSI (Relative Strength Index) and other technical signals
    - **Fibonacci Levels**: Key retracement and extension levels
    
    The color intensity in the chart indicates the confidence level - darker green shows stronger buy zones (at lower prices) and darker red shows stronger sell zones (at higher prices).
    """)
    
    st.subheader("About This Tool")
    st.markdown("""
    This valuation tool uses multiple methods to estimate a stock's intrinsic value, combined with technical analysis to identify optimal entry and exit points. It's designed to help investors make more informed decisions by providing a comprehensive analysis of a stock's value from both fundamental and technical perspectives.
    
    **Disclaimer**: This tool provides estimates based on the data you input. These estimates should not be considered as investment advice. Always do your own research or consult a financial advisor before making investment decisions.
    """)
    
st.markdown("---")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.caption("© 2025 NiveshLakshya")

with col2:
    st.markdown("""
    <div style="text-align: center;">
        <a href="mailto:somnathsingh044@gmail.com" target="_blank" style="margin-right: 15px; text-decoration: none;">
            <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="currentColor" viewBox="0 0 16 16">
                <path d="M.05 3.555A2 2 0 0 1 2 2h12a2 2 0 0 1 1.95 1.555L8 8.414.05 3.555ZM0 4.697v7.104l5.803-3.558L0 4.697ZM6.761 8.83l-6.57 4.027A2 2 0 0 0 2 14h12a2 2 0 0 0 1.808-1.144l-6.57-4.027L8 9.586l-1.239-.757Zm3.436-.586L16 11.801V4.697l-5.803 3.546Z"/>
            </svg>
        </a>
        <a href="https://www.linkedin.com/in/somnath-singh-384a83235" target="_blank" style="margin-right: 15px; text-decoration: none;">
            <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="currentColor" viewBox="0 0 16 16">
                <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
            </svg>
        </a>
        <a href="https://github.com/somnathsingh31/stock-valuation-python" target="_blank" style="text-decoration: none;">
            <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="currentColor" viewBox="0 0 16 16">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
            </svg>
        </a>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.caption("Data is for educational purposes only. This is not investment advice.")