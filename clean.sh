#!/bin/bash
# 删除空目录的脚本
 
dir_name=$1
loop=1
CMD="/usr/bin/find ${dir_name} -type d -empty "
 
${CMD} > dirlist.txt
 
while [ $(cat  dirlist.txt| wc -l) -gt  0  ]
do
	echo ------- Deleted Dir in Loop: ${loop} --------
	cat dirlist.txt
	${CMD} -exec rm -rf {} \; &>/dev/null 
	
	loop=$[loop + 1]
	${CMD} > dirlist.txt
	
done 
rm -rf dirlist.txt