# Get operating system name
uname_S := $(shell sh -c 'uname -s 2>/dev/null || echo not')

BINDIR = ../../bin

CC=gcc

OPT=-O3 -Wall -g -pg

ifeq ($(uname_S),Darwin)
	LDFLAGS	?= -framework OpenGL -framework GLUT
else
	LDFLAGS ?= -lGL -lGLU -lglut 
endif

all: $(BINDIR)/visAstrix

$(BINDIR)/visAstrix: visAstrix.o readfiles.o keyb.o disp.o
	$(CC) $(OPT) -o $@ $+ $(LDFLAGS)

visAstrix.o: visAstrix.c visAstrix.h
	$(CC) $(OPT) $(DEFS) -c visAstrix.c 

readfiles.o: readfiles.c visAstrix.h
	$(CC) $(OPT) $(DEFS) -c readfiles.c 

keyb.o: keyb.c visAstrix.h
	$(CC) $(OPT) $(DEFS) -c keyb.c 

disp.o: disp.c visAstrix.h
	$(CC) $(OPT) $(DEFS) -c disp.c 

clean:
	rm -f *~
	rm -f *.o 
	rm -f $(BINDIR)/visAstrix
