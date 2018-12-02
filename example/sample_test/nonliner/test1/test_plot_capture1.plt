set border lc rgb "black"
set grid lc rgb "#D8D8D8" lt 2
set key opaque box
#set yrang[-1:20]
set object 1 rect behind from screen 0,0 to screen 1,1 fc rgb "#FAFAFA" fillstyle solid

file = "test.dat"

plot file using 1:2   t "predict"  with lines linewidth 2
replot file using 1:3   t "Observation"  with lines linewidth 1 dt 5

replot file using 1:4   t "predict"  with lines linewidth 2
replot file using 1:5   t "Observation"  with lines linewidth 1 dt 5

replot file using 1:6   t "predict"  with lines linewidth 2
replot file using 1:7   t "Observation"  with lines linewidth 1 dt 5

set terminal png size 1048, 480
imgfile="images/test.png"
set output imgfile
pause 1
replot
