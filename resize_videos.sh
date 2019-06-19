#!/bin/bash

# resize FILE video to an OUT_HEIGHT,OUT_WIDTH size video saved in OUT
# CVU 2019

DAT

for i in {1..4}; do
  FILE="$DATA_DIR/orig/new_part$i.mp4"
  TMP="$DATA_DIR/scaled/new_part$i.mp4"
  OUT="$DATA_DIR/scaled_cropped/new_part$i.mp4"

  OUT_WIDTH=224
  OUT_HEIGHT=224

  # get the size of input video
  eval $(ffprobe -v error -of flat=s=_ -select_streams v:0 -show_entries stream=height,width ${FILE})
  IN_WIDTH=${streams_stream_0_width}
  IN_HEIGHT=${streams_stream_0_height}

  # get the difference between actual and desired size
  W_DIFF=$[ ${OUT_WIDTH} - ${IN_WIDTH} ]
  H_DIFF=$[ ${OUT_HEIGHT} - ${IN_HEIGHT} ]

  # let's take the shorter side, so the video will be at least as big as the desired size
  CROP_SIDE="n"
  if [ ${W_DIFF} -lt ${H_DIFF} ] ; then
    SCALE="-2:${OUT_HEIGHT}"
    CROP_SIDE="w"
  else
    SCALE="${OUT_WIDTH}:-2"
    CROP_SIDE="h"
  fi

  # then perform a first resizing
  ffmpeg -i ${FILE} -vf scale=${SCALE} ${TMP}

  # now get the temporary video size
  eval $(ffprobe -v error -of flat=s=_ -select_streams v:0 -show_entries stream=height,width ${TMP})
  IN_WIDTH=${streams_stream_0_width}
  IN_HEIGHT=${streams_stream_0_height}

  # calculate how much we should crop
  if [ "z${CROP_SIDE}" = "zh" ] ; then
    DIFF=$[ ${IN_HEIGHT} - ${OUT_HEIGHT} ]
    CROP="in_w:in_h-${DIFF}"
  elif [ "z${CROP_SIDE}" = "zw" ] ; then
    DIFF=$[ ${IN_WIDTH} - ${OUT_WIDTH} ]
    CROP="in_w-${DIFF}:in_h"
  fi

  # then crop
  ffmpeg -i ${TMP} -filter:v "crop=${CROP}" ${OUT}
done
