set encoding utf8
set border lc rgb "black"
set grid lc rgb "#D8D8D8" lt 2
set key opaque box
set object 1 rect behind from screen 0,0 to screen 1,1 fc rgb "#FAFAFA" fillstyle solid
set key right bottom

# smooth [unique, csplines, acsplines, bezier, sbezier]

plot 'accuracy.dat' using 1   t "train accuracy"  with lines linewidth 1 linecolor rgbcolor "#F5A9A9" dt 1
replot 'accuracy.dat' using 2  t "test accuracy" with lines linewidth 1 linecolor rgbcolor "#A9BCF5" dt 1
replot 'accuracy.dat' using 1  smooth bezier t "train"  with lines linewidth 2 linecolor rgbcolor "red"
replot 'accuracy.dat' using 2  smooth bezier t "test" with lines linewidth 2 linecolor rgbcolor "blue"

pause 10
reread
