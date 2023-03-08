#!/bin/bash

isort .
black --exclude='.*\/*(venv|static|test-results)\/*.*' .