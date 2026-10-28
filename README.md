# 🛡️ SupplyChain Guard AI

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An autonomous agent that turns a supply chain disruption into a dollar-quantified, costed rerouting decision.**
Built for the Nebius x NVIDIA Global AI Hackathon (Best Apps and Agents track).

## Quick Start (Local Execution)

1. Extract this zip file into `C:\SupplyChainGuard`
2. Open Command Prompt and run:
   ```cmd
   cd C:\SupplyChainGuard
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and insert your API keys:
   ```cmd
   copy .env.example .env
   ```
4. Run the Streamlit app:
   ```cmd
   streamlit run app_2.py
   ```
