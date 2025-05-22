
# 📊 NiveshLakshya

[![HuggingFace Spaces](https://img.shields.io/badge/Hosted%20on-HuggingFace-%23ff8a65)](https://huggingface.co/spaces/somnathsingh31/NiveshLakshya)

**NiveshLakshya** is a web-based stock valuation tool designed to perform **fundamental analysis** of companies, especially for the **Indian stock market**. It uses five classic valuation methods to estimate intrinsic value and generate detailed reports, including visual plots.

Built with **Streamlit** for ease of use, it's ideal for retail investors, financial analysts, and students looking to make informed decisions.

---

## 🎥 Tutorial Videos

To get started quickly, check out these walkthroughs:

1. 📊 **Using the Application**: [Watch Video](https://www.youtube.com/watch?v=O8WO64kL2uA)  
2. 📂 **Finding Financial Data**: [Watch Video](https://www.youtube.com/watch?v=sKDqJPivfF4)

---

## 💡 Tip

If financial data (like EPS, FCF, or ROE) fluctuates significantly year to year, it's advisable to **use a 5-year average** to get more stable and realistic valuation results.

---

## 🧮 Valuation Methods Used

This tool supports the following valuation techniques:

* **PE Valuation** – Based on earnings and adjusted P/E multiples
* **PB Valuation** – Uses book value and Return on Equity (ROE)
* **Dividend Discount Model (DDM)** – Projects future dividends and discounts them
* **Discounted Cash Flow (DCF)** – Forecasts free cash flows and discounts them
* **Graham Number** – Conservative formula using EPS and book value

Based on these, the app provides **intrinsic value estimates** and **comprehensive valuation reports** with supporting plots.

---

## 🌟 Key Features

* Supports **three investor types**: *Neutral*, *Optimistic*, and *Pessimistic*
* Generates visual and tabular comparisons of valuation outputs
* Final investment recommendation based on valuation confidence
* Simple and intuitive interface powered by Streamlit
* Publicly hosted on [Hugging Face Spaces](https://huggingface.co/spaces/somnathsingh31/NiveshLakshya)

---

## 📈 New in v2: Technical Analysis with Buy & Sell Zones

The upgraded version now includes **technical analysis** to complement fundamental valuation.

* Detects **buy and sell zones** using moving averages, Bollinger Bands, Fibonacci levels, and price clusters
* Visualizes **zones on price charts** with color-coded confidence levels (High, Medium, Low)
* Identifies market correction context and oversold conditions
* Generates a **comprehensive strategy** on when to start accumulating or reducing exposure
* Offers a **confidence score and textual reasoning** based on price-action and technical signals

This addition helps users make more **timing-aware decisions**, especially in volatile market phases.

---

## 🔎 Where to Find Financial Data

You can collect financial data manually using:

* [StockEdge](https://web.stockedge.com/) – Preferred for its user-friendly interface
* [Moneycontrol](https://www.moneycontrol.com/)

### Recommended: **StockEdge**

1. Search for a stock on [StockEdge](https://web.stockedge.com/)
2. Go to the **Fundamental** section
3. Under **Ratios**, explore the dropdown menus to find:

   * EPS (Earnings Per Share)
   * Book Value
   * Dividend
   * Free Cash Flow
   * ROE, Profit Margin, and other key metrics

---

## 🧠 How It Works

You enter basic financial data for a stock, choose your investor type (*Neutral*, *Optimistic*, or *Pessimistic*), and the app instantly calculates intrinsic values using multiple valuation models.

The app is built using Python and Streamlit, offering a seamless way to generate full valuation reports — including charts and recommendations — all from your browser.

---

## 📦 Installation (for local use)

```bash
git clone https://github.com/somnathsingh31/stock-valuation-python.git
cd stock-valuation-python
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚠️ Disclaimer

This tool is intended for **educational and informational purposes only**.
All outputs are based on standard models and the data you provide. It does **not constitute financial advice**, nor does it guarantee any investment outcome.
Always do your own research or consult a certified financial advisor before investing.

---