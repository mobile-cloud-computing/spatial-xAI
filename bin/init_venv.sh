#!/bin/bash
set -e

#init venv
python3.9 -m venv .xAI-Microservice

#Pip dependencies
.xAI-Microservice/bin/pip install --upgrade pip
.xAI-Microservice/bin/pip install -r requirements.txt
