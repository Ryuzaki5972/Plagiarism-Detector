CC = gcc
CFLAGS = -w -std=c99
LDFLAGS = -lm

TARGET = plagiarism_detector
SRC = plagiarism_detector.c
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJ) Output.txt PlagiarismReport.txt

.PHONY: all clean
