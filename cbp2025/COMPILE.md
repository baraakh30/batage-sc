# CORTEX Compilation Guide

## Prerequisites
- g++ compiler with C++17 support (MinGW-W64 on Windows)
- zlib library (`-lz`)
- CBP2025 framework library (already built in `lib/`)

## Quick Build (Recommended)

```bash
cd cbp2025
make clean
make
```

## Manual Compilation Steps

### Step 1: Clean Previous Builds
```bash
cd D:\uv\advarc2\cbp2025
Remove-Item -Force *.o, cbp.exe -ErrorAction SilentlyContinue
```

### Step 2: Build Framework Library (if needed)
```bash
cd lib
g++ -std=c++17 -O3 -I.. -I. -DGZSTREAM_NAMESPACE=gz -c cbp.cc
g++ -std=c++17 -O3 -I.. -I. -DGZSTREAM_NAMESPACE=gz -c uarchsim.cc
g++ -std=c++17 -O3 -I.. -I. -DGZSTREAM_NAMESPACE=gz -c gzstream.cc
ar rcs libcbp.a bp.o cache.o cbp.o gzstream.o my_value_predictor.o parameters.o resource_schedule.o uarchsim.o
cd ..
```

### Step 3: Compile CORTEX Predictor
```bash
g++ -std=c++17 -O3 -c cortex_predictor.cc -o cortex_predictor.o
```

### Step 4: Compile Interface
```bash
g++ -std=c++17 -O3 -c cond_branch_predictor_interface.cc -o cond_branch_predictor_interface.o
```

### Step 5: Link Executable
```bash
g++ -std=c++17 -O3 -L./lib -o cbp.exe cond_branch_predictor_interface.o cortex_predictor.o -lcbp -lz
```

## Testing

### Run with Sample Trace
```bash
.\cbp.exe sample_traces\int\sample_int_trace.gz
```

### Run with Full Trace
```bash
.\cbp.exe ..\data\branch\int\int_0_trace.gz
```

### Extract Branch Prediction Stats
```bash
.\cbp.exe sample_traces\int\sample_int_trace.gz 2>&1 | Select-String -Pattern "BRANCH|Misprediction|BrMisPKI"
```

## Switch Between Predictors

### To use BATAGE instead of CORTEX:

**Edit Makefile:**
```makefile
OBJ = cond_branch_predictor_interface.o batage_sc_predictor.o
DEPS = cbp.h cond_branch_predictor_interface.h batage_sc_predictor.h
```

**Edit cond_branch_predictor_interface.cc:**
```cpp
#include "batage_sc_predictor.h"
```

**Rebuild:**
```bash
make clean
make
```

### To switch back to CORTEX:

**Edit Makefile:**
```makefile
OBJ = cond_branch_predictor_interface.o cortex_predictor.o
DEPS = cbp.h cond_branch_predictor_interface.h cortex_predictor.h
```

**Edit cond_branch_predictor_interface.cc:**
```cpp
#include "cortex_predictor.h"
```

**Rebuild:**
```bash
make clean
make
```

## One-Line Commands (PowerShell)

### Build CORTEX
```powershell
cd D:\uv\advarc2\cbp2025; Remove-Item -Force *.o, cbp.exe -ErrorAction SilentlyContinue; g++ -std=c++17 -O3 -c cortex_predictor.cc -o cortex_predictor.o; g++ -std=c++17 -O3 -c cond_branch_predictor_interface.cc -o cond_branch_predictor_interface.o; g++ -std=c++17 -O3 -L./lib -o cbp.exe cond_branch_predictor_interface.o cortex_predictor.o -lcbp -lz
```

### Test
```powershell
cd D:\uv\advarc2\cbp2025; .\cbp.exe sample_traces\int\sample_int_trace.gz
```

## Compilation Flags Explained

- **-std=c++17**: Use C++17 standard
- **-O3**: Maximum optimization level
- **-L./lib**: Add lib/ directory to library search path
- **-lcbp**: Link with libcbp.a static library
- **-lz**: Link with zlib compression library
- **-DGZSTREAM_NAMESPACE=gz**: Define namespace for gzstream
- **-I.. -I.**: Add include paths for header files

## Expected Output

### Successful Compilation
```
cortex_predictor.o created
cond_branch_predictor_interface.o created
cbp.exe created (~960KB)
```

### Successful Run
```
EOF
WINDOW_SIZE = 1024
FETCH_WIDTH = 16
...
BRANCH PREDICTION MEASUREMENTS
...
```

## Troubleshooting

### Error: "cannot find -lz"
**Solution:** Install zlib development package

### Error: "undefined reference to gzstreambuf"
**Solution:** Rebuild lib with `-DGZSTREAM_NAMESPACE=gz`:
```bash
cd lib
make clean
make
cd ..
```

### Error: "Assertion failed: mInstr.mType != InstClass::undefInstClass"
**Solution:** Use correct trace file path:
```bash
.\cbp.exe sample_traces\int\sample_int_trace.gz
```
NOT: `.\cbp.exe sample_traces\int\INT-0.trace.gz` (wrong filename)

### Error: "has no member named 'notify_decode'"
**Solution:** Make sure you're using `cortex_predictor.h` include, not `batage_sc_predictor.h`

## Files Required for Compilation

### CORTEX Predictor
- `cortex_predictor.h` (677 lines)
- `cortex_predictor.cc` (1,154 lines)
- `cond_branch_predictor_interface.cc` (169 lines)

### Framework Headers
- `cbp.h`
- `lib/sim_common_structs.h`
- `lib/spdlog/spdlog.h`

### Framework Library
- `lib/libcbp.a` (contains 8 object files)

## Build Time

**Clean build:** ~10 seconds  
**Incremental build:** ~2 seconds
