file_name="include/Frame.h"
d1="/SP-SLAM/"
d2="/slamdoom/tmp/orbslam2/"

f1=${d1}${file_name}
f2=${d2}${file_name}

diff $f1 $f2 > /diff.txt
