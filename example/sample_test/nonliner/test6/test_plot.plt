set border lc rgb "black"
set grid lc rgb "#D8D8D8" lt 2
set key opaque box
#set yrang[-1:20]
set object 1 rect behind from screen 0,0 to screen 1,1 fc rgb "#FAFAFA" fillstyle solid
#set size ratio -1

unset key

file = "test.dat"

plot file using 1:2   t "predict"  with lines linewidth 2 lc "orange-red",\
"predict.dat" using 1:2   t "predict"  with lines linewidth 2 lc "orangered4",\
"predict_all.dat" using 1:2   t "predict_all"  with lines linewidth 2 lc "magenta",\
"prophecy.dat" using 1:2   t "prophecy"  with lines linewidth 2 lc "magenta"  dt 3

replot file using 1:3   t "Observation"  with lines linewidth 2 lc "web-blue",\
"predict.dat" using 1:3   t "predict"  with lines linewidth 2 lc "web-blue" dt 3,\
"predict_all.dat" using 1:3   t "predict_all"  with lines linewidth 2 lc "web-blue" dt 3

replot file using 1:4   t "predict"  with lines linewidth 2 lc "orange-red",\
"predict.dat" using 1:4   t "predict"  with lines linewidth 2  lc "orangered4",\
"predict_all.dat" using 1:4   t "predict_all"  with lines linewidth 2 lc "magenta",\
"prophecy.dat" using 1:2   t "prophecy"  with lines linewidth 2 lc "magenta"  dt 3

replot file using 1:5   t "Observation"  with lines linewidth 2 lc "web-blue",\
"predict.dat" using 1:5   t "Observation"  with lines linewidth 2 lc "web-blue" dt 3,\
"predict_all.dat" using 1:5   t "Observation"  with lines linewidth 2 lc "web-blue" dt 3

replot file using 1:6   t "predict"  with lines linewidth 2 lc "orange-red",\
"predict.dat" using 1:6   t "predict"  with lines linewidth 2  lc "orangered4",\
"predict_all.dat" using 1:6   t "predict_all"  with lines linewidth 2 lc "magenta",\
"prophecy.dat" using 1:2   t "prophecy"  with lines linewidth 2 lc "magenta"  dt 3

replot file using 1:7   t "Observation"  with lines linewidth 2 lc "web-blue",\
"predict.dat" using 1:7   t "Observation"  with lines linewidth 2 lc "web-blue" dt 3,\
"predict_all.dat" using 1:7   t "Observation"  with lines linewidth 2 lc "web-blue" dt 3


#set terminal png
#set out "image1.png"
#replot

#set terminal windows
#set output

pause 3
reread