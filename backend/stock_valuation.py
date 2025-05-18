import logging
from datetime import datetime
from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pydantic import BaseModel, Field, ValidationError, field_validator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialData(BaseModel):
    eps: float
    book_value: float
    dividend: float
    free_cash_flow: float
    eps_growth_rate: float = Field(..., ge=0, le=1)
    roe: float = Field(..., ge=0, le=1)
    debt_to_equity: float
    profit_margin: float = Field(..., ge=0, le=1)
    current_ratio: float
    pe_ratio_industry: float
    industry: str = "Other"

    @field_validator("*", mode="before")
    @classmethod
    def check_finite_values(cls, v):
        if isinstance(v, (int, float)) and not np.isfinite(v):
            raise ValueError("Value must be finite")
        return v

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        if denominator == 0 or not np.isfinite(denominator) or not np.isfinite(numerator):
            return default
        result = numerator / denominator
        if not np.isfinite(result): # Catch division results like inf/-inf if somehow missed
            return default
        return result
    except Exception as e:
        logger.warning(f"Division error: {numerator}/{denominator}. Error: {e}")
        return default

def discounted_terminal_value(cash_flow: float, growth: float, rate: float, years: int) -> float:
    try:
        if not all(np.isfinite(x) for x in [cash_flow, growth, rate]) or not isinstance(years, int) or years < 0:
            logger.warning("Non-finite input or invalid years for discounted terminal value calculation")
            return 0.0
        
        if rate <= growth:
            logger.warning("Terminal value calculation: discount rate must be greater than growth rate")
            return 0.0
        
        terminal_value_at_year_n = (cash_flow * (1 + growth)) / (rate - growth)
        present_terminal_value = terminal_value_at_year_n / ((1 + rate) ** years)
        
        if not np.isfinite(present_terminal_value):
            logger.warning("Non-finite result in discounted terminal value calculation")
            return 0.0
        return present_terminal_value
    except Exception as e:
        logger.error(f"Error in discounted terminal value calculation: {e}")
        return 0.0

class StockValuation:
    def __init__(self, company_name: str, current_price: float):
        try:
            if not isinstance(company_name, str) or not company_name.strip():
                raise ValueError("Company name must be a non-empty string")
            
            if not isinstance(current_price, (int, float)) or current_price <= 0 or not np.isfinite(current_price):
                raise ValueError(f"Current price must be a positive finite number, got {current_price}")
            
            self.company_name = company_name.strip()
            self.current_price = float(current_price)
            self.data: Dict[str, Any] = {}
            self.valuation_results: Dict[str, Dict[str, float]] = {}
        except Exception as e:
            logger.error(f"StockValuation initialization error: {e}")
            raise

    def set_financial_data(self, **kwargs) -> bool:
        try:
            self.data = FinancialData(**kwargs).dict()
            return True
        except ValidationError as e:
            error_msgs = [f"{error.get('loc', ['UnknownField'])[0]}: {error.get('msg', 'Unknown error')}" for error in e.errors()]
            logger.error(f"Invalid financial data: {', '.join(error_msgs)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in set_financial_data: {e}")
            return False

    def pe_valuation(self) -> float:
        try:
            if not self.data:
                logger.warning("Financial data not set for PE valuation")
                return 0.0
            
            eps_growth_percentage = self.data["eps_growth_rate"] * 100
            growth_pe = 8.0 + 2.0 * eps_growth_percentage
            
            if self.data["debt_to_equity"] > 1.0:
                growth_pe *= 0.9
            if self.data["current_ratio"] < 1.5:
                growth_pe *= 0.95
            
            company_industry = self.data.get("industry", "Other")
            
            if company_industry == "Information Technology":
                if self.data["eps_growth_rate"] > 0.15:
                    growth_pe *= 1.2
                if self.data["profit_margin"] > 0.25:
                    growth_pe *= 1.1
            elif company_industry == "Banking":
                growth_pe *= 0.85
                if self.data["roe"] > 0.18:
                    growth_pe *= 1.15
            elif company_industry == "FMCG":
                growth_pe *= 1.1
                if self.data["profit_margin"] > 0.15:
                    growth_pe *= 1.05
            elif company_industry == "Pharmaceuticals":
                growth_pe *= 1.05
                if self.data["debt_to_equity"] > 0.8:
                    growth_pe *= 0.9
            elif company_industry == "Automobile":
                growth_pe *= 0.9
                if self.data["debt_to_equity"] > 0.6:
                    growth_pe *= 0.9
            elif company_industry == "Energy":
                growth_pe *= 0.85
            elif company_industry == "Telecommunications":
                growth_pe *= 0.95
                if self.data["debt_to_equity"] > 1.2:
                    growth_pe *= 0.85
            elif company_industry == "Manufacturing":
                growth_pe *= 0.92
                if self.data["profit_margin"] > 0.12:
                    growth_pe *= 1.15
            
            intrinsic_calculated_pe = self.data["eps"] * growth_pe
            intrinsic_industry_pe = self.data["eps"] * self.data["pe_ratio_industry"]
            
            final_intrinsic_value: float
            if company_industry in ["Information Technology", "Pharmaceuticals"]:
                final_intrinsic_value = 0.65 * intrinsic_calculated_pe + 0.35 * intrinsic_industry_pe
            elif company_industry in ["Banking", "Energy", "Telecommunications"]:
                final_intrinsic_value = 0.55 * intrinsic_calculated_pe + 0.45 * intrinsic_industry_pe
            else: # Covers "Other" and any unspecified industries
                final_intrinsic_value = 0.60 * intrinsic_calculated_pe + 0.40 * intrinsic_industry_pe
            
            if final_intrinsic_value < 0: final_intrinsic_value = 0.0

            difference = safe_divide((final_intrinsic_value - self.current_price), self.current_price, 0.0) * 100
            
            self.valuation_results["PE"] = {
                "intrinsic_value": final_intrinsic_value,
                "difference": difference
            }
            return final_intrinsic_value
        except Exception as e:
            logger.error(f"Error in PE valuation: {e}")
            self.valuation_results["PE"] = {"intrinsic_value": 0.0, "difference": -100.0 if self.current_price > 0 else 0.0}
            return 0.0

    def pb_valuation(self) -> float:
        try:
            if not self.data:
                logger.warning("Financial data not set for PB valuation")
                return 0.0
                
            pb_multiplier = self.data["roe"] * 5.5
            if self.data["debt_to_equity"] > 0.5:
                pb_multiplier *= max(0, (1 - (self.data["debt_to_equity"] - 0.5) * 0.2)) # Ensure multiplier doesn't go negative
                
            intrinsic = self.data["book_value"] * pb_multiplier
            if intrinsic < 0: intrinsic = 0.0
            
            difference = safe_divide((intrinsic - self.current_price), self.current_price, 0.0) * 100
            
            self.valuation_results["PB"] = {
                "intrinsic_value": intrinsic,
                "difference": difference
            }
            return intrinsic
        except Exception as e:
            logger.error(f"Error in PB valuation: {e}")
            self.valuation_results["PB"] = {"intrinsic_value": 0.0, "difference": -100.0 if self.current_price > 0 else 0.0}
            return 0.0

    def dividend_discount_model(self, required_return: float = 0.15, terminal_growth: float = 0.04) -> float:
        try:
            if not self.data:
                logger.warning("Financial data not set for DDM valuation")
                return 0.0
                
            if not all(np.isfinite(x) for x in [required_return, terminal_growth]):
                logger.warning("Non-finite input values for DDM valuation parameters")
                return 0.0
                
            if required_return <= terminal_growth:
                logger.warning("DDM: Required return must be greater than terminal growth")
                self.valuation_results["DDM"] = {"intrinsic_value": 0.0, "difference": -100.0 if self.current_price > 0 else 0.0}
                return 0.0
                
            if self.data["dividend"] <= 0 or self.data["eps"] <= 0:
                logger.info("DDM: Dividend or EPS is zero or negative, DDM value is 0.")
                self.valuation_results["DDM"] = {"intrinsic_value": 0.0, "difference": -100.0 if self.current_price > 0 else 0.0}
                return 0.0
                
            payout_ratio = safe_divide(self.data["dividend"], self.data["eps"])
            retention_ratio = max(0.0, min(1.0, 1.0 - payout_ratio))
            
            implied_growth = self.data["roe"] * retention_ratio
            # Use a blend or conservative estimate for growth, capped at a reasonable high-growth rate
            growth_rate_phase1 = min(max(implied_growth, self.data["eps_growth_rate"] * 0.8, 0.0), 0.25) # Take higher of implied or 80% of EPS growth, cap at 25%
            
            projected_value = 0.0
            current_dividend = self.data["dividend"]
            
            projection_years = 5
            for i in range(1, projection_years + 1):
                current_dividend *= (1 + growth_rate_phase1)
                projected_value += safe_divide(current_dividend, ((1 + required_return) ** i))
                
            terminal_val = discounted_terminal_value(current_dividend, terminal_growth, required_return, projection_years)
            projected_value += terminal_val
            
            if projected_value < 0: projected_value = 0.0

            difference = safe_divide((projected_value - self.current_price), self.current_price, 0.0) * 100
            
            self.valuation_results["DDM"] = {
                "intrinsic_value": projected_value,
                "difference": difference
            }
            return projected_value
        except Exception as e:
            logger.error(f"Error in DDM valuation: {e}")
            self.valuation_results["DDM"] = {"intrinsic_value": 0.0, "difference": -100.0 if self.current_price > 0 else 0.0}
            return 0.0

    def dcf_valuation(self, required_return: float = 0.15, terminal_growth: float = 0.04) -> float:
        try:
            if not self.data:
                logger.warning("Financial data not set for DCF valuation")
                return 0.0
                
            if not all(np.isfinite(x) for x in [required_return, terminal_growth]):
                logger.warning("Non-finite input values for DCF valuation parameters")
                return 0.0
                
            if required_return <= terminal_growth:
                logger.warning("DCF: Required return must be greater than terminal growth")
                self.valuation_results["DCF"] = {"intrinsic_value": 0.0, "difference": -100.0 if self.current_price > 0 else 0.0}
                return 0.0
                
            if self.data["free_cash_flow"] <= 0:
                logger.info("DCF: Free Cash Flow is zero or negative, DCF value is 0.")
                self.valuation_results["DCF"] = {"intrinsic_value": 0.0, "difference": -100.0 if self.current_price > 0 else 0.0}
                return 0.0
                
            current_fcf = self.data["free_cash_flow"]
            # Use a scaled EPS growth as proxy for FCF growth, capped
            growth_rate_phase1 = min(max(self.data["eps_growth_rate"] * 0.9, 0.0), 0.25) 
            
            projected_value = 0.0
            projection_years = 5
            for i in range(1, projection_years + 1):
                current_fcf *= (1 + growth_rate_phase1)
                projected_value += safe_divide(current_fcf, ((1 + required_return) ** i))
                
            terminal_val = discounted_terminal_value(current_fcf, terminal_growth, required_return, projection_years)
            projected_value += terminal_val

            if projected_value < 0: projected_value = 0.0
            
            difference = safe_divide((projected_value - self.current_price), self.current_price, 0.0) * 100
            
            self.valuation_results["DCF"] = {
                "intrinsic_value": projected_value,
                "difference": difference
            }
            return projected_value
        except Exception as e:
            logger.error(f"Error in DCF valuation: {e}")
            self.valuation_results["DCF"] = {"intrinsic_value": 0.0, "difference": -100.0 if self.current_price > 0 else 0.0}
            return 0.0

    def graham_number(self) -> float:
        try:
            if not self.data:
                logger.warning("Financial data not set for Graham Number calculation")
                return 0.0
                
            if self.data["eps"] <= 0 or self.data["book_value"] <= 0:
                logger.info("Graham: EPS or Book Value is zero or negative, Graham Number is 0.")
                self.valuation_results["Graham"] = {"intrinsic_value": 0.0, "difference": -100.0 if self.current_price > 0 else 0.0}
                return 0.0
            
            graham_val = 0.0
            try:
                graham_val = np.sqrt(22.5 * self.data["eps"] * self.data["book_value"])
                if not np.isfinite(graham_val): graham_val = 0.0
            except Exception as e:
                logger.error(f"Error calculating square root for Graham Number: {e}")
                self.valuation_results["Graham"] = {"intrinsic_value": 0.0, "difference": -100.0 if self.current_price > 0 else 0.0}
                return 0.0
            
            # Custom adjustment based on ROE
            if self.data["roe"] > 0.15: # If ROE is above 15%
                graham_val *= (1 + (self.data["roe"] - 0.15)) # Boost Graham value
            elif self.data["roe"] < 0.08 and self.data["roe"] > 0: # If ROE is low but positive
                 graham_val *= (1 - (0.08 - self.data["roe"])*2) # Penalize for low ROE

            if graham_val < 0: graham_val = 0.0 
            
            difference = safe_divide((graham_val - self.current_price), self.current_price, 0.0) * 100
            
            self.valuation_results["Graham"] = {
                "intrinsic_value": graham_val,
                "difference": difference
            }
            return graham_val
        except Exception as e:
            logger.error(f"Error in Graham Number calculation: {e}")
            self.valuation_results["Graham"] = {"intrinsic_value": 0.0, "difference": -100.0 if self.current_price > 0 else 0.0}
            return 0.0

    def intrinsic_value_calculator(self, profile: Optional[str] = "neutral") -> Dict[str, Any]:
        try:
            if not self.data:
                logger.error("Financial data not set for intrinsic value calculation")
                return {"error": "Financial data not set", "intrinsic_value": 0.0, "current_price": self.current_price, "potential": 0.0}
            
            profile_str = profile.lower() if isinstance(profile, str) else "neutral"
            
            base_weights = {
                "optimistic": {"PE": 0.25, "PB": 0.10, "DDM": 0.05, "DCF": 0.45, "Graham": 0.15},
                "neutral":    {"PE": 0.30, "PB": 0.15, "DDM": 0.05, "DCF": 0.40, "Graham": 0.10},
                "pessimistic":  {"PE": 0.15, "PB": 0.25, "DDM": 0.15, "DCF": 0.20, "Graham": 0.25}
            }

            weights = base_weights.get(profile_str, base_weights["neutral"]).copy()
            
            company_industry = self.data.get("industry", "Other")
            has_dividends = self.data.get("dividend", 0) > 0
            
            # Industry-specific weight adjustments
            if company_industry == "Banking":
                weights["PB"] = min(weights["PB"] * 1.5, 0.35)  
                weights["DDM"] = weights["DDM"] * 1.3 if has_dividends else weights["DDM"] * 0.5
                weights["DCF"] = max(weights["DCF"] * 0.8, 0.20)
            elif company_industry == "Information Technology":
                weights["DCF"] = min(weights["DCF"] * 1.3, 0.55)  
                weights["PB"] = weights["PB"] * 0.7  
                if not has_dividends and "DDM" in weights:
                    ddm_weight = weights.pop("DDM", 0)
                    weights["DCF"] += ddm_weight * 0.7
                    weights["PE"] += ddm_weight * 0.3
            elif company_industry == "FMCG":
                weights["PE"] = min(weights["PE"] * 1.2, 0.40)
                weights["DDM"] = weights["DDM"] * 1.3 if has_dividends else weights["DDM"] * 0.7
                weights["Graham"] = max(weights["Graham"] * 0.8, 0.05)
            

            methods_to_run = {
                "PE": self.pe_valuation, "PB": self.pb_valuation,
                "DDM": self.dividend_discount_model, "DCF": self.dcf_valuation,
                "Graham": self.graham_number
            }
            
            for name, method_func in methods_to_run.items():
                if name not in self.valuation_results or self.valuation_results[name].get("intrinsic_value", -1) <= 0: # Rerun if not present or non-positive
                    if name == "DDM" and not has_dividends: 
                         continue
                    method_func()
                
                if self.valuation_results.get(name, {}).get("intrinsic_value", -1) <= 0 and name in weights:
                    weights.pop(name)

            valid_method_results = {k: v for k, v in self.valuation_results.items() 
                                    if v.get("intrinsic_value", 0) > 0 and k in weights}
                        
            if not valid_method_results:
                logger.warning("No valid valuation methods with positive intrinsic values and weights available.")
                return {
                    "intrinsic_value": 0.0, "current_price": self.current_price, "potential": -100.0 if self.current_price > 0 else 0.0,
                    "individual_valuations": {k: round(v.get("intrinsic_value",0), 2) for k,v in self.valuation_results.items()},
                    "weights": {}, "warning": "No valid valuation methods available or all resulted in zero/negative values."
                }
            
            current_weights = {k: weights[k] for k in valid_method_results}
            total_weight = sum(current_weights.values())
            
            if total_weight <= 0:
                logger.warning("Total weight for valid methods is zero or negative after filtering.")
                return {
                    "intrinsic_value": 0.0, "current_price": self.current_price, "potential": -100.0 if self.current_price > 0 else 0.0,
                    "individual_valuations": {k: round(v["intrinsic_value"], 2) for k, v in valid_method_results.items()},
                    "weights": {}, "warning": "Total weight of applicable methods is zero."
                }
                
            normalized_weights = {k: v / total_weight for k, v in current_weights.items()}
            
            final_intrinsic = sum(valid_method_results[k]["intrinsic_value"] * w 
                                  for k, w in normalized_weights.items())
                        
            potential_upside = safe_divide((final_intrinsic - self.current_price), self.current_price, 0.0) * 100

            return {
                "intrinsic_value": round(final_intrinsic, 2),
                "current_price": self.current_price,
                "potential": round(potential_upside, 2),
                "individual_valuations": {k: round(v["intrinsic_value"], 2) for k, v in valid_method_results.items()},
                "weights": {k: round(w * 100) for k, w in normalized_weights.items()}, # Show weights as percentages
                "industry": company_industry
            }
        except Exception as e:
            logger.error(f"Critical error in intrinsic_value_calculator: {e}", exc_info=True)
            return {
                "error": f"Calculation error: {str(e)}", "current_price": self.current_price,
                "intrinsic_value": 0.0, "potential": -100.0 if self.current_price > 0 else 0.0
            }

    def get_recommendation(self, result: Dict[str, Any]) -> str:
        try:
            if "error" in result and result["error"]:
                return f"Unable to provide recommendation: {result['error']}"
            if result.get("warning"):
                 return f"Recommendation cautious: {result['warning']}"
                
            potential = result.get("potential", 0.0)
            
            if not np.isfinite(potential):
                return "Recommendation unavailable due to non-finite potential."

            if potential > 30: return "Strong Buy: Significantly undervalued"
            if potential > 15: return "Buy: Moderately undervalued"
            if potential > 5: return "Accumulate: Slightly undervalued"
            if potential >= -5: return "Hold: Fairly valued" 
            if potential > -15: return "Reduce: Slightly overvalued"
            return "Sell: Significantly overvalued" 
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "Unable to generate recommendation due to an unexpected error."

    def plot_valuation_comparison(self):
        try:
            valid_methods = {k: v for k, v in self.valuation_results.items() 
                          if v.get("intrinsic_value", 0) > 0}
                          
            if not valid_methods:
                st.warning("No valid valuation methods to display in chart")
                return
                
            methods = list(valid_methods.keys())
            values = [valid_methods[m]["intrinsic_value"] for m in methods]

            fig, ax = plt.figure(figsize=(12, 7)), plt.gca()
            
            try:
                bars = ax.bar(methods, values, color='#3498db')
                ax.axhline(y=self.current_price, color='r', linestyle='--', linewidth=2, 
                         label=f'Current Price ₹{self.current_price:,.2f}')

                for bar in bars:
                    h = bar.get_height()
                    if np.isfinite(h):
                        ax.text(bar.get_x() + bar.get_width() / 2., h + max(h * 0.01, 1), 
                              f'₹{h:,.0f}', ha='center')

                ax.set_title(f'Valuation Comparison for {self.company_name}')
                ax.set_ylabel('Stock Price (₹)')
                plt.xticks(rotation=15)
                ax.legend()
                plt.tight_layout()
                
                st.pyplot(fig)
            except Exception as e:
                logger.error(f"Error creating matplotlib visualization: {e}")
                st.error("Failed to create visualization")
        except Exception as e:
            logger.error(f"Error in plot_valuation_comparison: {e}")
            st.error("Unable to generate valuation comparison plot")

    def generate_report(self) -> str:
        try:
            result = self.intrinsic_value_calculator()
            
            if "error" in result:
                return f"Error generating report: {result['error']}"
                
            recommendation = self.get_recommendation(result)
            industry = self.data.get("industry", "Other")

            # Use string Templates for better formatting
            from string import Template
            from textwrap import dedent
            
            # Top summary
            top_summary = (
                f"Intrinsic Value: ₹{result['intrinsic_value']:.2f}\n"
                f"Potential: {result['potential']:.2f}%\n"
                f"Recommendation: {recommendation}\n\n"
            )

            # Report header and summary
            template = dedent('''
            =========================================================
            FUNDAMENTAL VALUATION REPORT FOR $company_name
            =========================================================
            Date: $date
            Industry: $industry

            SUMMARY:
            --------
            Current Market Price: ₹$current_price
            Calculated Intrinsic Value: ₹$intrinsic_value
            Potential Upside/Downside: $potential%
            Recommendation: $recommendation

            VALUATION BREAKDOWN:
            --------------------
            ''')
            
            # Fill in the template
            report = Template(template).substitute(
                company_name=self.company_name.upper(),
                date=datetime.now().strftime('%Y-%m-%d'),
                industry=industry,
                current_price=f"{self.current_price:.2f}",
                intrinsic_value=f"{result['intrinsic_value']:.2f}",
                potential=f"{result['potential']:.2f}",
                recommendation=recommendation
            )
            
            # Add valuation methods
            if not result.get("individual_valuations"):
                report += "No valid valuation methods available.\n"
            else:
                for method, value in result["individual_valuations"].items():
                    if method in self.valuation_results:
                        diff = self.valuation_results[method]["difference"]
                        weight = result["weights"].get(method, 0)
                        status = "Undervalued" if diff > 0 else "Overvalued"
                        report += f"{method} Valuation: ₹{value:.2f} ({status} by {abs(diff):.2f}%, Weight: {weight}%)\n"

            # Add financial metrics section
            report += dedent('''
            
            FINANCIAL METRICS USED:
            ----------------------
            ''')
            
            if not self.data:
                report += "Financial data not available.\n"
            else:
                try:
                    metrics = [
                        f"EPS: ₹{self.data['eps']:.2f}",
                        f"Book Value: ₹{self.data['book_value']:.2f}",
                        f"Dividend: ₹{self.data['dividend']:.2f}",
                        f"Free Cash Flow: ₹{self.data['free_cash_flow']:.2f}",
                        f"EPS Growth Rate: {self.data['eps_growth_rate'] * 100:.2f}%",
                        f"ROE: {self.data['roe'] * 100:.2f}%",
                        f"Debt/Equity: {self.data['debt_to_equity']:.2f}",
                        f"Profit Margin: {self.data['profit_margin'] * 100:.2f}%",
                        f"Current Ratio: {self.data['current_ratio']:.2f}",
                        f"Industry P/E: {self.data['pe_ratio_industry']:.2f}"
                    ]
                    report += "\n".join(metrics) + "\n"
                except Exception as e:
                    logger.error(f"Error displaying financial metrics: {e}")
                    report += "Error displaying some financial metrics.\n"

            # Add analysis notes section
            report += dedent('''
            
            ANALYSIS NOTES:
            --------------
            ''')
            
            notes = []
            
            # General valuation comments
            if result["potential"] > 15:
                notes.append("- The stock appears significantly undervalued.")
            elif result["potential"] < -15:
                notes.append("- The stock appears significantly overvalued.")
                
            # Financial health comments
            if self.data:
                if self.data.get("debt_to_equity", 0) > 1:
                    notes.append("- High debt levels are a concern.")

                if self.data.get("roe", 0) > 0.15:
                    notes.append("- Strong ROE indicates efficient management.")
                else:
                    notes.append("- ROE is below optimal levels.")

                if self.data.get("eps_growth_rate", 0) > 0.15:
                    notes.append("- Strong earnings growth supports premium valuation.")
                    
                if self.data.get("current_ratio", 0) < 1:
                    notes.append("- Low current ratio indicates potential liquidity issues.")
                    
                if self.data.get("profit_margin", 0) < 0.05:
                    notes.append("- Low profit margins may impact long-term sustainability.")
            
            # Add the notes if any
            if notes:
                report += "\n".join(notes) + "\n"
            
            # Industry-specific analysis section
            report += dedent(f'''
            
            INDUSTRY-SPECIFIC INSIGHTS ({industry}):
            -------------------------------------
            ''')
            
            industry_insights = []
            
            if industry == "Information Technology":
                industry_insights = [
                    "- IT companies typically command higher multiples due to scalability and lower capital intensity."
                ]
                if self.data.get("eps_growth_rate", 0) < 0.1:
                    industry_insights.append("- Growth rate below 10% is concerning for IT companies in the current environment.")
                if self.data.get("profit_margin", 0) > 0.2:
                    industry_insights.append("- High profit margins indicate strong competitive positioning in the IT sector.")
                industry_insights.append("- IT valuations are strongly influenced by future growth prospects rather than current book value.")
                
            elif industry == "Banking":
                industry_insights = [
                    "- Banking valuations are particularly sensitive to interest rate changes and credit cycles."
                ]
                if self.data.get("debt_to_equity", 0) > 10:
                    industry_insights.append("- High leverage is normal for banks but warrants monitoring in the current economic climate.")
                industry_insights.append("- Price-to-Book ratio is especially relevant for banking valuations.")
                if self.data.get("roe", 0) < 0.1:
                    industry_insights.append("- ROE below 10% is a concern for banking stocks.")
                    
            elif industry == "FMCG":
                industry_insights = [
                    "- FMCG companies typically trade at premium multiples due to brand value and stable earnings."
                ]
                if self.data.get("profit_margin", 0) < 0.08:
                    industry_insights.append("- Profit margins below industry average may indicate pricing pressure or inefficient operations.")
                industry_insights.append("- Dividend stability is particularly valued in the FMCG sector.")
                
            elif industry == "Pharmaceuticals":
                industry_insights = [
                    "- Pharmaceutical valuations consider both current earnings and R&D pipeline potential."
                ]
                if self.data.get("eps_growth_rate", 0) > 0.15:
                    industry_insights.append("- Strong growth indicates successful product development or market expansion.")
                industry_insights.append("- Regulatory approvals and patent expirations can significantly impact pharma valuations.")
                
            elif industry == "Automobile":
                industry_insights = [
                    "- Auto sector valuations are cyclical and sensitive to economic conditions."
                ]
                if self.data.get("debt_to_equity", 0) > 0.8:
                    industry_insights.append("- Debt levels are higher than ideal for navigating industry downturns.")
                industry_insights.append("- Capital expenditure requirements impact free cash flow in the auto sector.")
                
            elif industry == "Energy":
                industry_insights = [
                    "- Energy company valuations are influenced by commodity price cycles and regulatory environment."
                ]
                if self.data.get("dividend", 0) > 0:
                    industry_insights.append("- Dividend yield provides support for valuation during industry downturns.")
                industry_insights.append("- Asset replacement value is an important consideration for energy companies.")
                
            elif industry == "Telecommunications":
                industry_insights = [
                    "- Telecom valuations are supported by infrastructure assets and recurring revenue streams."
                ]
                if self.data.get("debt_to_equity", 0) > 1.5:
                    industry_insights.append("- High debt levels require careful monitoring given capital-intensive nature of the business.")
                industry_insights.append("- Spectrum assets and subscriber metrics are key valuation drivers beyond financial statements.")
                
            elif industry == "Manufacturing":
                industry_insights = [
                    "- Manufacturing valuations tend to be cyclical and asset-intensive."
                ]
                if self.data.get("current_ratio", 0) < 1.2:
                    industry_insights.append("- Current ratio below optimal levels may indicate working capital challenges.")
                if self.data.get("roe", 0) > 0.15:
                    industry_insights.append("- ROE above 15% demonstrates efficient asset utilization relative to industry norms.")
                
            elif industry == "Other":
                industry_insights = [
                    "- Company-specific factors and peer comparison are key for valuation analysis.",
                    "- Consider industry positioning and competitive advantages beyond the metrics."
                ]
            
            # Add industry insights
            if industry_insights:
                report += "\n".join(industry_insights) + "\n"
            
            # Add disclaimer
            report += dedent('''
            
            DISCLAIMER:
            -----------
            This analysis is based on historical data and standard valuation
            models. It does not guarantee future performance. Always do
            your own research or consult a financial advisor before investing.
            =========================================================
            ''')
            
            return top_summary + report
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {str(e)}"


def create_stock_valuation(company_name: str, current_price: Union[float, str]) -> Optional[StockValuation]:
    try:
        if not isinstance(company_name, str) or not company_name.strip():
            logger.error("Invalid company name")
            return None
            
        if isinstance(current_price, str):
            try:
                current_price = float(current_price.replace(',', '').strip())
            except ValueError:
                logger.error(f"Invalid price format: {current_price}")
                return None
                
        if not isinstance(current_price, (int, float)) or current_price <= 0 or not np.isfinite(current_price):
            logger.error(f"Invalid current price: {current_price}")
            return None
            
        return StockValuation(company_name, current_price)
    except Exception as e:
        logger.error(f"Error creating stock valuation: {e}")
        return None