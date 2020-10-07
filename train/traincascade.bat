opencv_traincascade.exe -data Haar_xml -vec pos.vec -bg neg_img.txt   -numStages 15 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 3450   -numNeg 9000 -w 24 -h 24 -mode ALL
pause