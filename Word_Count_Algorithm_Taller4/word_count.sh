#!/usr/bin/bash

function ctrl_c () {
	echo "[!] Exiting ... Bye🤫🧏🗿"
}

trap ctrl_c SIGINT

content=$(cat "$2")



function normalize () {
	normalized_content=$(echo "$content" | tr -d '[:punct:]' | tr -s ' ')
	echo "$normalized_content"
}

function count_words () {
	normalized=$(normalize)
	echo -n "$normalized" | wc -w
}

while getopts ":nw" opt; do
	case ${opt} in
		n)
			echo "[+] El contenido del archivo normalizado es:" 
			normalize
			;;
		w)
			echo "[!] El archivo $2 contiene $(count_words) parabras." 
			;;						
	esac
done