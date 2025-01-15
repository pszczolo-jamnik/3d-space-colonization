# 3d-space-colonization
🦀 Optimized, parallel implementation of space colonization algorithm in 3D\
Accepts PLY point cloud as input and writes PLY output with custom vertex attributes

Mostly inspired by [@jasonwebb](https://github.com/jasonwebb) article and implementation

## Pictures
![](https://github.com/user-attachments/assets/0f02b4fe-42e3-4ad7-a4bb-8217bccada1c)
<img src="https://github.com/user-attachments/assets/a86c0ab4-cbb5-4501-ad71-418a38020906" width="580" />
<img src="https://github.com/user-attachments/assets/73a8ca61-cca8-4050-9a97-a5c98239839b" width="580" />

# Build
1. Install Rust https://www.rust-lang.org/tools/install
2. In repo root, run `cargo build --release`
3. Executable will be in `target/release` dir

# Usage
```
space-colonization.exe [OPTIONS] --file-in <FILE_IN> --file-out <FILE_OUT> --iterations <ITERATIONS>
--attraction-distance <ATTRACTION_DISTANCE> --kill-distance <KILL_DISTANCE>
--segment-length <SEGMENT_LENGTH> <--origin-random <N>|--origin-min <AXIS>>

Options:
      --file-in <FILE_IN>
          Input PLY file
      --file-out <FILE_OUT>
          Output PLY file
      --iterations <ITERATIONS>
      --attraction-distance <ATTRACTION_DISTANCE>
          Only nodes within this distance around an attractor can be associated with that attractor.
          Large attraction distances mean smoother and more subtle branch curves, but at a performance cost
      --kill-distance <KILL_DISTANCE>
          An attractor may be removed if one or more nodes are within this distance around it
      --segment-length <SEGMENT_LENGTH>
          The distance between nodes as the network grows.
          Larger values mean better performance, but choppier and sharper branch curves
      --origin-random <N>
          Start from N random points
      --origin-min <AXIS>
          Start from minimum along (x | y | z)
      --random <R>
          Randomize growth 0.1 - 1.0
  -h, --help
          Print help
  -V, --version
          Print version
```
