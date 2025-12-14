NAME        :=	linear-regression
PYTHON      :=	python3

VENV_DIR    :=	.venv
ACTIVATE    :=	$(VENV_DIR)/bin/activate

TRAIN       :=	train.py
ESTIMATE    :=	predict.py
EVAL        :=	evaluate.py

RM          :=	rm -f
OUT_FILES	:=	plot.png training.gif model.json

.DEFAULT_GOAL	:=	all

.PHONY: all
all: train estimate evaluate

.PHONY: venv
venv: $(VENV_DIR)

$(VENV_DIR):
	@printf " Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo "  "

.PHONY: install
install: venv
	@printf " Installing matplotlib..."
	@. $(ACTIVATE) && pip install --upgrade pip setuptools wheel > /dev/null 2>&1
	@. $(ACTIVATE) && pip install matplotlib > /dev/null 2>&1
	@echo "  "

.PHONY: train
train: install
	@echo " Training model..."
	@. $(ACTIVATE) && $(PYTHON) $(TRAIN) 10000 0.1

.PHONY: estimate
estimate: install
	@echo " Check estimation..."
	@. $(ACTIVATE) && $(PYTHON) $(ESTIMATE)

.PHONY: evaluate
evaluate: install
	@echo " Evaluating model..."
	@. $(ACTIVATE) && $(PYTHON) $(EVAL)

.PHONY: clean
clean:
	@printf "󰃢 Cleaning generated files..."
	@$(RM) $(OUT_FILES)
	@echo "  "

.PHONY: fclean
fclean: clean
	@printf " Removing environment and artifacts..."
	@$(RM) -r $(VENV_DIR) __pycache__ 2>/dev/null || true
	@echo "  "

.PHONY: re
re: fclean all
