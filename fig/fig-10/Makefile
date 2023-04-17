PY=python
EPS2PDF=ps2pdf
CROP=pdfcrop

SRC := $(wildcard *.py)
SRC := $(filter-out zplot.py, $(SRC))
EPS = $(patsubst %.py,%.eps,$(SRC))
PDF = $(patsubst %.py,%.pdf,$(SRC))

all: $(EPS) $(PDF)

%.eps: %.py 
	$(PY) $< eps

%.pdf: %.eps
	$(EPS2PDF) $<
	$(CROP) $@ $@
	rm -rf $<

fig:
# mv -f *.pdf ../../../fig/
	mv -f *.pdf ../
clean: 
	rm -rf *.eps *.pdf
