/usr/bin/time -v ./mpich_master-50.sh fk 8 2>&1 | tee log/part50/fk_8.log
/usr/bin/time -v  ./mpich_master-50.sh ytb 8 2>&1 | tee log/part50/ytb_8.log
/usr/bin/time -v  ./mpich_master-50.sh soc 8 2>&1 | tee log/part50/soc_8.log
/usr/bin/time -v  ./mpich_master-50.sh LJ 8 2>&1 | tee log/part50/LJ_8.log
/usr/bin/time -v  ./mpich_master-50.sh com 8 2>&1 | tee log/part50/com_8.log
/usr/bin/time -v  ./mpich_master-50.sh twt 8 2>&1 | tee log/part50/twt_8.log
