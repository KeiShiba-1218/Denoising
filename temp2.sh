for res in 4 8 16 32 64 128 256
do
    for channels in 16 32 64 128 256
    do
        python temp.py $res $channels
    done
done