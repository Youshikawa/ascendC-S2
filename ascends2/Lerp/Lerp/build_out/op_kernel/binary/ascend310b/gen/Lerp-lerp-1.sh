#!/bin/bash
echo "[Ascend310B1] Generating Lerp_2c5332bfea4d2b6c519a7a414ae75aa9 ..."
opc $1 --main_func=lerp --input_param=/home/HwHiAiUser/guodong/workspace/ascend-s2/Lerp/Lerp/build_out/op_kernel/binary/ascend310b/gen/Lerp_2c5332bfea4d2b6c519a7a414ae75aa9_param.json --soc_version=Ascend310B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/Lerp_2c5332bfea4d2b6c519a7a414ae75aa9.json ; then
  echo "$2/Lerp_2c5332bfea4d2b6c519a7a414ae75aa9.json not generated!"
  exit 1
fi

if ! test -f $2/Lerp_2c5332bfea4d2b6c519a7a414ae75aa9.o ; then
  echo "$2/Lerp_2c5332bfea4d2b6c519a7a414ae75aa9.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating Lerp_2c5332bfea4d2b6c519a7a414ae75aa9 Done"
