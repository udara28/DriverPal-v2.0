# DriverPal-v2.0
This repository contains the project developed for Xilinx Adaptive Computing Contest based on ZCU104
For the project description please goto [hackster.io project page](https://www.hackster.io/akronzippy/driverpal-v2-1dfe2b)

## Build Instructions

This project contains a desktop version and a embedded version (tested on Xilinx FPGA board ZCU104)

### Building Desktop Version ###

```
bash build_app.sh
./a.out
```

### Building Embedded Version ###

```
mkdir build && cd build
cmake ..
make
```

**We noticed a link error when trying to build standalone using above commands. Use the source code in Vitis_In_Depth_Tutorial/Machine_Learning/Introduction/03-Basic/Module_7/app/test to compile if this is the case**

## Video Demonstration

[Video Link](https://youtu.be/5MW2A_vnDx4)


## Contributors

* [Udara De Silva](https://udaradesilva.com)
* [John T Vorhies](https://www.linkedin.com/in/johnvorhies/)

## Acknowledgment

This work is inspired by following works
* [Vitis-In-Depth-Tutorial](https://github.com/Xilinx/Vitis-In-Depth-Tutorial)
* [CarND-Advanced-Lane-Lines](https://github.com/ajsmilutin/CarND-Advanced-Lane-Lines)
