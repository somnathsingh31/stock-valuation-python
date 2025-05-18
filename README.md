# 📈 Stock Valuation App

A web-based stock valuation tool for performing **fundamental analysis** of companies, with a focus on **Indian markets**. This app uses classic valuation methods like:

- **Discounted Cash Flow (DCF)**
- **Dividend Discount Model (DDM)**
- **Price-to-Earnings (P/E)**
- **Price-to-Book (P/B)**
- **Graham Number**

Built with **Streamlit** for interactive analysis and easy use by retail investors, financial analysts, and students.

---

## 🚀 Features

- Input key financial data and instantly get intrinsic valuation
- Choose investor sentiment: *Optimistic*, *Neutral*, or *Pessimistic*
- Visual comparison of valuation methods
- Full valuation report with investment recommendation
- Deployed using [Hugging Face Spaces](https://huggingface.co/spaces)

---

## 🧠 How It Works

The backend logic is implemented in Python and uses `pydantic` for robust financial data validation. Each valuation method is tailored to suit Indian market characteristics such as conservative multipliers and risk-adjusted discount rates.

The Streamlit frontend provides an intuitive interface to enter data, select sentiment profile, and generate a full valuation report.

---

## 📦 Installation

```bash
git clone https://github.com/your-username/stock-valuation.git
cd stock-valuation
pip install -r requirements.txt
streamlit run app/app.py
```


## Disclaimer

⚠️ **Disclaimer**

This tool is intended for **educational and informational purposes only**. 
The valuation outcomes are based on static valuation models and user-provided inputs. 
It **does not guarantee future stock performance** and should **not be used as financial advice**. 

Please conduct your own due diligence or consult a licensed financial advisor before making any investment decisions.