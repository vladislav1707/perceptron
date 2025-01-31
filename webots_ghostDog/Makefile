# Компилятор и флаги
CXX = g++
CXXFLAGS = -std=c++17 -Wall -g
OPENMP_FLAGS = -fopenmp

# Пути к Webots
WEBOTS_HOME ?= "C:/Program Files/Webots"
WEBOTS_INCLUDE = $(WEBOTS_HOME)/include/controller/cpp
WEBOTS_LIB = $(WEBOTS_HOME)/lib/controller

# Включаем пути к заголовочным файлам и библиотекам
INCLUDES = -I"$(WEBOTS_INCLUDE)"
LIBS = -L"$(WEBOTS_LIB)" -lController -lCppController -lgomp

# Цель сборки
TARGET = main.exe
SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)

# Проверка существования файлов
CHECK_FILES := $(wildcard $(SRCS))
ifeq ($(CHECK_FILES),)
    $(error Source file $(SRCS) not found!)
endif

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LIBS) $(OPENMP_FLAGS)

clean:
	del /F $(TARGET) *.o 2>nul || true

rebuild: clean all

.PHONY: all clean rebuild