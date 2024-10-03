# xAI Service

The xAI methods (LIME, SHAP, Occlusion Sensivity) are provided as microservices which is accessible by FastAPI (REST API). 

## Prerequisites

- Python 3.8.10
- Docker installed if you wish to containerize the application

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/Mrasinthe/xAI.git
```

Create and activate environment Variable:

```bash
python3 -m venv venv
source venv/bin/activate
```

Change directory to the cloned repository:

```bash
cd xAI
```

Install python requirements:

```bash
pip install -r requirements.txt
```

## Run the application locally

```bash
python3 main.py
```

## Deploy using Docker

Go to the 'xai_docker' folder and run the command below

```bash
sudo docker-compose up --build
```

