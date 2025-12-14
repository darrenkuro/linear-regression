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

$(VENV_DIR):
	@echo "🐍 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)

.PHONY: venv
venv: $(VENV_DIR)

.PHONY: install
install: venv
	@echo "📦 Installing matplotlib..."
	. $(ACTIVATE) && pip install --upgrade pip setuptools wheel
	. $(ACTIVATE) && pip install matplotlib

.PHONY: train
train: install
	@echo "🚀 Training model..."
	. $(ACTIVATE) && $(PYTHON) $(TRAIN) 10000 0.1

.PHONY: estimate
estimate: install
	. $(ACTIVATE) && $(PYTHON) $(ESTIMATE)

.PHONY: evaluate
evaluate: install
	@echo "📊 Evaluating model..."
	. $(ACTIVATE) && $(PYTHON) $(EVAL)

.PHONY: clean
clean:
	@echo "🧹 Cleaning generated files..."
	$(RM) $(OUT_FILES)

.PHONY: fclean
fclean: clean
	@echo "🔥 Removing environment and artifacts..."
	$(RM) -r $(VENV_DIR) __pycache__ 2>/dev/null || true

.PHONY: re
re: fclean all
