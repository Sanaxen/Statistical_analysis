ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.mp4
ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.flv
ffmpeg -y -i "images/test_%%04d.png" -an -r 8  -pix_fmt rgb24 -vf "setpts=2.0*PTS"  -vf "scale=500:-1" -f gif bbb.gif