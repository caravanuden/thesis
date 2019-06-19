#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for i in {0..10..2}
  do
     echo "Welcome $i times"
 done

for i in {1..825}
  do
    a=$((15*i))
    cp raiders/images/part_6/frame_0_img_0.jpg "raiders/images/test/frame_$a_img_$i.jpg"
  done

for i in {825..840}
  do
    a=$((15*i))
    echo "$a"
    cp cp "raiders/images/part_6/frame_$a_img_$i.jpg" "raiders/images/test/frame_$a_img_$i.jpg"
  done

for i in {840..1000}
  do
    a=$((15*i))
    echo "$a"
    cp raiders/images/part_6/frame_0_img_0.jpg "raiders/images/test/frame_$a_img_$i.jpg"
  done
