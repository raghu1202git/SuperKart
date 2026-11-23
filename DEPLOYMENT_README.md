# SuperKart Deployment Guide

## Completed artifacts
- `train_superkart.py`: feature engineering, model tuning, comparison, and model serialization
- `backend_files/superkart_model.joblib`: selected Random Forest pipeline
- `backend_files/model_comparison.csv`: candidate model metrics
- `app.py`: Flask API with `/health`, `/v1/predict`, and `/v1/predictbatch`
- `streamlit_app.py`: interactive frontend
- `backend_Dockerfile`, `frontend_Dockerfile`: container definitions

## Local validation
```bash
cd /Users/raghu/Python_Projects/Multi_Agentic_AI
.venv/bin/python train_superkart.py
.venv/bin/python app.py
```
Open `http://localhost:7860/health` and `http://localhost:8501` after starting Streamlit.

## Docker
```bash
docker network create superkart-network
docker build -f backend_Dockerfile -t superkart-backend .
docker run -d --name superkart-backend --network superkart-network -p 7860:7860 superkart-backend
docker build -f frontend_Dockerfile -t superkart-frontend .
docker run -d --name superkart-frontend --network superkart-network -p 8501:8501 superkart-frontend
```
Docker Desktop must be running. The frontend connects to `superkart-backend:7860` on the shared Docker network.

## Required deployment credentials
1. **GitHub account**: required to host code and create a Codespace.
2. **GitHub Personal Access Token (classic)**: required only if you push with HTTPS and password authentication. Set scope `repo` for a private repository.
3. **Public Codespace port**: set port `7860` to Public for external Flask API access. Set port `8501` Public if the frontend should be opened externally.

No Hugging Face key is required for this implementation. No OpenAI or LangSmith key is required. No database password is required.

## Optional production credentials
- API authentication secret, if you add bearer authentication
- Cloud provider account or container registry credentials for Azure, AWS, GCP, or Docker Hub

Never place credentials in source code or commit `.env` files to GitHub.
