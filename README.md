# CLM #

## Requirements ##
* Linux (testing Ubuntu 14.04 and Arch Linux)
* OpenCV 3.1 (2.4 is also supported if replace link options in `premake5.lua`)
* premake 5

## Build and Run ##
Edit `premake5.lua` and `CASCADE_FILE` variable in `src/main.cpp`, `src/train.cpp` and `yukiti.cpp` for your environment.

```
premake5 gmake
cd build
make
./bin/release/main
./bin/release/yukiti
```
