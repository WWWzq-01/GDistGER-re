/usr/bin/time -v ./mpich_master-100.sh fk 8 2>&1 | tee log/part100/fk_8.log
/usr/bin/time -v  ./mpich_master-100.sh ytb 8 2>&1 | tee log/part100/ytb_8.log
/usr/bin/time -v  ./mpich_master-100.sh soc 8 2>&1 | tee log/part100/soc_8.log
/usr/bin/time -v  ./mpich_master-100.sh LJ 8 2>&1 | tee log/part100/LJ_8.log
/usr/bin/time -v  ./mpich_master-100.sh com 8 2>&1 | tee log/part100/com_8.log
/usr/bin/time -v  ./mpich_master-100.sh twt 8 2>&1 | tee log/part100/twt_8.log
