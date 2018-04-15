# CLM #
"Constrained Local Models" based on "Mastering OpenCV Chapter6".

I refactored all sources and add warping demo.
Tracking accuracy is not so high, but more readable code than original to study CLM.

## Requirements ##
* Linux (testing Ubuntu 14.04 and Arch Linux)
* OpenCV 3.1 (2.4 is also supported if replaced link options in `premake5.lua`)
* premake 5 or CMake

## Build ##
Edit `premake5.lua` for your environment.

```
premake5 gmake
cd build
make
```
or
```
mkdir build
cd build
cmake ..
make
```

Following commands can be executed in `build` directory.

## Run with trained model ##
```
./bin/release/main
```

To configure cascade path, set new path with `--cascade` argument,
and CLM model also can be changed with `--clm` (default model is trained using `helen` dataset).
Web camera will be used by default, and image can be used with `--image`.

<img src="https://raw.githubusercontent.com/takiyu/CLM/master/screenshots/main_lena.png" width="360px">

## Train ##
Now, `MUCT` dataset is available to train (`helen` mode is broken).
Please download `MUCT` dataset, expand it and execute following command (Change paths).

```
./bin/release/train --out OUTPUT/DIR --muct_image_dir YOUR/muct/jpg/ --muct_lm_file YOUR/muct/muct-landmarks/muct76-opencv.csv
```

### Visualized shape ###
<img src="https://raw.githubusercontent.com/takiyu/CLM/master/screenshots/train_shape.gif" width="280px">

### Visualized patch ###
<img src="https://raw.githubusercontent.com/takiyu/CLM/master/screenshots/train_patch.png" width="280px">

### Visualized detector ###
<img src="https://raw.githubusercontent.com/takiyu/CLM/master/screenshots/train_detector.png" width="280px">

## Warping demo 1 ##
`yukiti` is a demo program which tracks your face using web camera and warps Japanese bill.

```
./bin/release/yukiti
```

<img src="https://raw.githubusercontent.com/takiyu/CLM/master/screenshots/yukiti.png" width="700px">

## Warping demo 2 ##
`yukiti2` is also a demo program.
This replaces your face with bill's one.

```
./bin/release/yukiti2
```
