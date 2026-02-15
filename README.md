# LOOKBACK OPTION PRICING: ANALYTICAL VS. MONTE CARLO

This project comes from my Undergraduate Thesis (TFG). The goal was to build a tool to price Floating Strike Lookback Puts, comparing the math formulas (Analytical) with a computer simulation (Monte Carlo).

---

## **WHAT I WORKED ON**

- **Handling large data:** Simulating 100,000 paths with 10,000 steps each is a lot for a standard computer. I used Batch Vectorization in Python to process the data in chunks, so it runs fast without crashing the RAM.
- **The Pricing Gap:** I used the project to visualize why the Monte Carlo price is always a bit lower than the theoretical one (lookback bias). It's a great way to see the difference between continuous math and discrete computer steps.
- **Risk (Delta):** I also added a script to calculate the Delta, showing how the option price reacts when the stock price moves.

## **PROJECT STRUCTURE**

- `lookback_pricing_mc.py`: The Python code with all the calculations.
- `docs/`: [Download Full Thesis (PDF)](./docs/Final%20Year%20Thesis%20–%20Valuation%20of%20Lookback%20Options.pdf) — _Note: The theoretical background is written in Catalan as part of my Undergraduate Thesis (TFG)._

## **RESULTS**

> **Analytical Price:** ~14.29  
> **Monte Carlo Price:** ~14.17 (100k sims)
