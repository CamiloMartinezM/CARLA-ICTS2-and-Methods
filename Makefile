#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = carla-ci3p-pgmpy
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python${PYTHON_VERSION}
VENV_PATH = $$HOME/.virtualenvs/$(PROJECT_NAME)/bin
SETUP_ENVIRONMENT_VARIABLES_SCRIPT = setup_envars.sh

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install --upgrade pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	@make setup_envars  # Automatically call setup_envars after installing requirements
	
## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff check .

## Format source code with ruff
.PHONY: format
format:
	ruff format .
	ruff check --fix .

## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	@make setup_envars  # Automatically call setup_envars after creating the environment

## Setup environment variables for virtualenv
.PHONY: setup_envars
setup_envars:
	@if [ ! -d "$(VENV_PATH)" ]; then \
		echo -e "\033[31m✘\033[0m \033[1mERROR\033[0m: Virtual environment '$(PROJECT_NAME)' not found in ~/.virtualenvs/"; \
		exit 1; \
	fi
	@cp "$(SETUP_ENVIRONMENT_VARIABLES_SCRIPT)" "$(VENV_PATH)/postactivate"
	@echo -e '#!/bin/bash\nunset LD_LIBRARY_PATH\necho -e "\033[34m🛈\033[0m \033[1mINFO\033[0m: Resetting LD_LIBRARY_PATH"' > "$(VENV_PATH)/predeactivate"
	@chmod +x "$(VENV_PATH)/postactivate" "$(VENV_PATH)/predeactivate"
	@echo -e "\033[34m🛈\033[0m \033[1mINFO\033[0m: Environment variable setup scripts have been created and made executable."


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
