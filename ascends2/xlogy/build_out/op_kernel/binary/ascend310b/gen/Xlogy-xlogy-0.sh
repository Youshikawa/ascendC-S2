#!/bin/bash
echo "[Ascend310B1] Generating Xlogy_04a7dddd80dbe3c8bc9be3a031af2a47 ..."
opc $1 --main_func=xlogy --input_param=/home/HwHiAiUser/yangzhichuan_test/xlogy_fix/xlogy/build_out/op_kernel/binary/ascend310b/gen/Xlogy_04a7dddd80dbe3c8bc9be3a031af2a47_param.json --soc_version=Ascend310B1 --output=$2 --impl_mode="" --simplified_key_mode=0 --op_mode=dynamic

if ! test -f $2/Xlogy_04a7dddd80dbe3c8bc9be3a031af2a47.json ; then
  echo "$2/Xlogy_04a7dddd80dbe3c8bc9be3a031af2a47.json not generated!"
  exit 1
fi

if ! test -f $2/Xlogy_04a7dddd80dbe3c8bc9be3a031af2a47.o ; then
  echo "$2/Xlogy_04a7dddd80dbe3c8bc9be3a031af2a47.o not generated!"
  exit 1
fi
echo "[Ascend310B1] Generating Xlogy_04a7dddd80dbe3c8bc9be3a031af2a47 Done"
