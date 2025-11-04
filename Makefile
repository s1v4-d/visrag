SHELL := /bin/bash
.ONESHELL:

PROJECT_DIR := $(PWD)
PROJECT_NAME := vlmrag
HOST_UID ?= $(shell id -u)
HOST_GID ?= $(shell id -g)
COMMIT_SHA := $(shell git rev-parse --short HEAD || echo dev)
BUILD_TIMESTAMP := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")

.PHONY: help build-docker start-docker stop-docker logs logs-all shell root lint fmt test

help:
	@echo "Targets: build-docker | start-docker | stop-docker | logs | logs-all | shell | root | lint | fmt | test"

.env:
	@echo "PROJECT_DIR=$(PROJECT_DIR)" > .env
	@echo "PROJECT_NAME=$(PROJECT_NAME)" >> .env
	@echo "HOST_UID=$(HOST_UID)" >> .env
	@echo "HOST_GID=$(HOST_GID)" >> .env
	@echo "COMMIT_SHA=$(COMMIT_SHA)" >> .env
	@echo "BUILD_TIMESTAMP=$(BUILD_TIMESTAMP)" >> .env
	@echo "OPENAI_API_KEY=replace_me" >> .env
	@echo "QDRANT_URL=http://qdrant:6333" >> .env
	@echo "SERVICE_PORT=8080" >> .env

build-docker: .env
	docker compose --env-file .env -f dev-env/docker-compose.yaml -p $(PROJECT_NAME) build

start-docker: .env
	docker compose --env-file .env -f dev-env/docker-compose.yaml -p $(PROJECT_NAME) up -d

stop-docker:
	docker compose --env-file .env -f dev-env/docker-compose.yaml -p $(PROJECT_NAME) down
	docker system prune -a -f --volumes

logs:
	docker compose --env-file .env -f dev-env/docker-compose.yaml -p $(PROJECT_NAME) logs -f api

logs-all:
	docker compose --env-file .env -f dev-env/docker-compose.yaml -p $(PROJECT_NAME) logs -f --tail=200

shell:
	- docker exec -it $(PROJECT_NAME)-api /bin/sh -lc "bash || sh"

root:
	- docker exec -u root -it $(PROJECT_NAME)-api /bin/sh -lc "bash || sh"

lint:
	docker exec $(PROJECT_NAME)-api /bin/sh -lc "/opt/venv/bin/ruff check src/ api/ --config /app/pyproject.toml"

fmt:
	docker exec $(PROJECT_NAME)-api /bin/sh -lc "/opt/venv/bin/ruff format src/ api/ --config /app/pyproject.toml"

test:
	docker exec $(PROJECT_NAME)-api /bin/sh -lc "/opt/venv/bin/pytest -q"
