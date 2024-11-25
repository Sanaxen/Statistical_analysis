bind "Close" "if (GPVAL_TERM eq \'wxt\') bind \'Close\' \'\'; exit gnuplot; else bind \'Close\' \'\'; exit"
set datafile separator ","

set encoding utf8
set border lc rgb "black"
set grid lc rgb "#D8D8D8" lt 2
set key opaque box
set object 1 rect behind from screen 0,0 to screen 1,1 fc rgb "#FAFAFA" fillstyle solid
set key right top

# smooth [unique, csplines, acsplines, bezier, sbezier]

plot 'lingam_loss.dat' using 1   t "Residual "  with lines linewidth 1 linecolor rgbcolor "#F5A9A9" dt 1
replot 'lingam_loss.dat' using 2  t "Independence" with lines linewidth 1 linecolor rgbcolor "#A9BCF5" dt 1
replot 'lingam_loss.dat' using 1  smooth bezier t "residual"  with lines linewidth 2 linecolor rgbcolor "#f39800"
replot 'lingam_loss.dat' using 2  smooth bezier t "independence" with lines linewidth 2 linecolor rgbcolor "#0068b7"

pause 10
reread
