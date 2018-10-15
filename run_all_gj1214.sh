for filename in `ls $HOME/Research/Planets/GJ1214/data/raw/ch2/wand_did_it/`
do
	echo $filename
	python Phase\ Curve\ Exoplanet\ TSO\ -\ Photometric\ Extraction\ Pipeline.py	--aor_dir $filename \
											--channel ch2 \
											--planet_name GJ1214 \
											--data_dir $HOME/Research/Planets/GJ1214/data/raw/ch2/wand_did_it/ \
											-sd ExtractedData \
											-pd /Research/Planets/ \
											-ds /data/raw \
											-dt /big/ \
                                                	                                -ff bcd \
                                                        	                        -uf bunc \
                                                                	                -m median \
                                                                        	        -t Spitzer \
                                                                                	-ou electrons
	# mv /Users/jfraine/Research/Planets/GJ1214/data/raw/ch2/big/$filename \
	# /Users/jfraine/Research/Planets/GJ1214/data/raw/ch2/big_done/
done
