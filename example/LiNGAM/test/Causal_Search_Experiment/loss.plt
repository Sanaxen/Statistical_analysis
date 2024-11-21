bind "Close" "if (GPVAL_TERM eq \'wxt\') bind \'Close\' \'\'; exit gnuplot; else bind \'Close\' \'\'; exit"

if (exist("n")==0 || n<0) n=0 #•Ï”‚Ì‰Šú‰»

if (n==0 ) set terminal png
if (n==0 ) set output '../Causal_Search_Experiment/loss.png'
cd '../../DDS2/Work'

set encoding utf8
set border lc rgb "black"
set grid lc rgb "#D8D8D8" lt 2
#set key opaque
set key box
set object 1 rect behind from screen 0,0 to screen 1,1 fc rgb "#FAFAFA" fillstyle solid
set key left top
set datafile separator ","

# smooth [unique, csplines, acsplines, bezier, sbezier]

plot 'lingam_loss.dat' using 1   t "residual error"  with lines linewidth 1 linecolor rgbcolor "#F5A9A9" dt 1 \
, 'lingam_loss.dat' using 2  t "residual independence" with lines linewidth 1 linecolor rgbcolor "#A9BCF5" dt 1 \
, 'lingam_loss.dat' using 3  t "loss" with lines linewidth 1 linecolor rgbcolor "#3cb371" dt 1 \
, 'lingam_loss.dat' using 4  t "vest" with lines linewidth 1 linecolor rgbcolor "#b0c4de" dt 1 \
, 'lingam_loss.dat' using 1  smooth bezier t "Residual"  with lines linewidth 2 linecolor rgbcolor "#f39800" \
, 'lingam_loss.dat' using 2  smooth bezier t "independence" with lines linewidth 2 linecolor rgbcolor "#0068b7" \
, 'lingam_loss.dat' using 3  smooth bezier t "Loss" with lines linewidth 2 linecolor rgbcolor "#006400" \
, 'lingam_loss.dat' using 4  smooth bezier t "Vest" with lines linewidth 2 linecolor rgbcolor "#0000cd" \


set terminal windows
set output
if( n == 1 ) { n=0;
}  else {
	n=1
}
#pause 10
#reread
