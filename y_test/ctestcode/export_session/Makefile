


CROSS_COMPILE = 
LD = $(CROSS_COMPILE)g++
CC = $(CROSS_COMPILE)g++
TARGET = export_session

SRCS = $(wildcard ./src/*.cpp)
OBJS := $(SRCS:.cpp=.o)

INCLUDES =  -I ./inc
LIBS = -L ./lib -lpthread
CCFLAGS = -g -Wall -O0

%.o : %.cpp
	$(CC) -o $@ $(INCLUDES) $(CCFLAGS) -c $<  


$(TARGET) : $(OBJS)
	$(CC) -o $@ $^ $(LIBS)

clean:
	rm $(OBJS) $(TARGET)

