if [ $# != 2 ]
then
    echo "Usage: sh run_distribution_ascend.sh [DEVICE_NUM] [CONFIG_PATH]"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo $1
    else
        echo "$(realpath -m ${PWD}/$1)"
    fi
}

CONFIG_PATH=$(get_real_path $2)

if [ ! -f $CONFIG_PATH ]
then
    echo "error: CONFIG_PATH=$CONFIG_PATH is not a file"
exit 1
fi

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

export RANK_SIZE=$1

cd $BASE_PATH
mpirun --allow-run-as-root -n $RANK_SIZE python ../train.py --config_path=$CONFIG_FILE --device_num=$RANK_SIZE > log.txt 2>&1 &