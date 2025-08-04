#!/bin/bash
set -e

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt
