all:
	g++ -g main.cpp -o main.exe -L C:\boost_1_85_0\build\lib -I C:\boost_1_85_0\build\include -lSDL2 -lSDL2main -Wall