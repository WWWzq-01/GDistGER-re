/usr/bin/time -v ./mpich_master.sh fk 8 2>&1 | tee log/fk_8.log
/usr/bin/time -v  ./mpich_master.sh ytb 8 2>&1 | tee log/ytb_8.log
/usr/bin/time -v  ./mpich_master.sh soc 8 2>&1 | tee log/soc_8.log
/usr/bin/time -v  ./mpich_master.sh LJ 8 2>&1 | tee log/LJ_8.log
/usr/bin/time -v  ./mpich_master.sh com 8 2>&1 | tee log/com_8.log
/usr/bin/time -v  ./mpich_master.sh twt 8 2>&1 | tee log/twt_8.log
