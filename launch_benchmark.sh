#!/bin/bash
set -xe

function main {
    # set common info
    source oob-common/common.sh
    init_params $@
    fetch_device_info
    set_environment

    # requirements
    pip uninstall -y transformers tokenizers
    pip install -U transformers accelerate
    python setup.py install

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        if [ "${model_name}" == "jmamou-ddpm-ema-flowers-64-distil-2layers" ];then
            model_arch=' --model_name_or_path jmamou/ddpm-ema-flowers-64-distil-2layers --model_type ddpm '
        elif [ "${model_name}" == "jmamou-ddpm-ema-flowers-64-distil-4layers" ];then
            model_arch=' --model_name_or_path jmamou/ddpm-ema-flowers-64-distil-4layers --model_type ddpm '
        elif [ "${model_name}" == "anton-l-ddpm-ema-flowers-64" ];then
            model_arch=' --model_name_or_path anton-l/ddpm-ema-flowers-64 --model_type ddpm '
        elif [ "${model_name}" == "CompVis-stable-diffusion-v1-4" ];then
            model_arch=' --model_name_or_path CompVis/stable-diffusion-v1-4 '
        elif [ "${model_name}" == "runwayml-stable-diffusion-v1-5" ];then
            model_arch=' --model_name_or_path runwayml/stable-diffusion-v1-5 '
        elif [ "${model_name}" == "stabilityai-stable-diffusion-2-1" ];then
            model_arch=' --model_name_or_path stabilityai/stable-diffusion-2-1 --dpm_solver '
        else
            echo "[Error] Not enable such model: $model_name"
            exit 1
        fi
        # cache
        if [ $batch_size -le 0 ];then
            batch_size=1
        fi
        python inference.py --device ${device} \
            ${model_arch} \
            --channels_last ${channels_last} \
            --num_iter 3 --num_warmup 1 \
            --precision=${precision} \
            --per_device_eval_batch_size ${batch_size} \
            ${addtion_options}
        #
        for batch_size in ${batch_size_list[@]}
        do
            if [ $batch_size -le 0 ];then
                batch_size=32
            fi
            # clean workspace
            logs_path_clean
            echo ${batch_size}
            # generate launch script for multiple instance
            if [ "${OOB_USE_LAUNCHER}" == "1" ] && [ "${device}" != "cuda" ];then
                generate_core_launcher
            else
                generate_core
            fi
            # launch
            echo -e "\n\n\n\n Running..."
            cat ${excute_cmd_file} |column -t > ${excute_cmd_file}.tmp
            mv ${excute_cmd_file}.tmp ${excute_cmd_file}
            source ${excute_cmd_file}
            echo -e "Finished.\n\n\n\n"
            # collect launch result
            collect_perf_logs
        done
    done
}

# run
function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        # instances
        if [ "${device}" != "cuda" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        else
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
        fi
        printf " ${OOB_EXEC_HEADER} \
            python inference.py --device ${device} \
                ${model_arch} \
                --channels_last ${channels_last} \
                --precision=${precision} \
                --per_device_eval_batch_size ${batch_size} \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

function generate_core_launcher {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        printf "python -m oob-common.launch --enable_jemalloc \
                    --core_list $(echo ${device_array[@]} |sed 's/;.//g') \
                    --log_file_prefix rcpi${real_cores_per_instance} \
                    --log_path ${log_dir} \
                    --ninstances ${#device_array[@]} \
                    --ncore_per_instance ${real_cores_per_instance} \
            inference.py --device ${device} \
                ${model_arch} \
                --channels_last ${channels_last} \
                --precision=${precision} \
                --per_device_eval_batch_size ${batch_size} \
                ${addtion_options} \
        > /dev/null 2>&1 &  \n" |tee -a ${excute_cmd_file}
        break
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# download common files
rm -rf oob-common && git clone https://github.com/intel-sandbox/oob-common.git

# Start
main "$@"
