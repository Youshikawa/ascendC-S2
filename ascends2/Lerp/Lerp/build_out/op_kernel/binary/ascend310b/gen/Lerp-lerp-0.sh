#!/bin/bash
echo "[Ascend310B1] Generating Lerp_c2d76805c5cc5d3b92dd6208d8173af8 ..."
opc $1 --main_func=lerp --input_param=/home/HwHiAiUser/guodong/workspace/ascend-s2/Lerp/Lerp/build_out/op_kernel/binary/ascend310b/gen/Lerp_c2d76805c5cc5d3b92dd6208d8173af8_param.json --soc_version=Ascend310B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/Lerp_c2d76805c5cc5d3b92dd6208d8173af8.json ; then
  echo "$2/Lerp_c2d76805c5cc5d3b92dd6208d8173af8.json not generated!"
  exit 1
fi

if ! test -f $2/Lerp_c2d76805c5cc5d3b92dd6208d8173af8.o ; then
  echo "$2/Lerp_c2d76805c5cc5d3b92dd6208d8173af8.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating Lerp_c2d76805c5cc5d3b92dd6208d8173af8 Done"
