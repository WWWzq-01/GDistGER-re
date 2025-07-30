#!/bin/bash
while read -r ip; do 
    echo $ip 
done < hosts
