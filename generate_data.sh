#!/bin/bash
set -e

mkdir -p logs

GENERATE_RAW=1
GENERATE_VERL=1

# generate raw data
if [[ $GENERATE_RAW -ne 0 ]]; then
    echo "Generating RLVR data..."

    python3 -u -m data_generation.generate data_generation/configs/rl/interventional.yaml > logs/rl_intv_gen.out 2> logs/rl_intv_gen.err &
    pid1=$!
    python3 -u -m data_generation.generate data_generation/configs/rl/observational.yaml > logs/rl_obs_gen.out 2> logs/rl_obs_gen.err &
    pid2=$!
    python3 -u -m data_generation.generate data_generation/configs/rl/counterfactual.yaml > logs/rl_counter_gen.out 2> logs/rl_counter_gen.err &
    pid3=$!

    wait $pid1
    echo "Interventional (PID $pid1) completed."

    wait $pid2
    echo "Observational (PID $pid2) completed."

    wait $pid3
    echo "Counterfactual (PID $pid3) completed."

    echo "Generating SFT data..."

    python3 -u -m data_generation.generate data_generation/configs/sft/interventional.yaml > logs/sft_intv_gen.out 2> logs/sft_intv_gen.err &
    pid1=$!
    python3 -u -m data_generation.generate data_generation/configs/sft/observational.yaml > logs/sft_obs_gen.out 2> logs/sft_obs_gen.err &
    pid2=$!
    python3 -u -m data_generation.generate data_generation/configs/sft/counterfactual.yaml > logs/sft_counter_gen.out 2> logs/sft_counter_gen.err &
    pid3=$!

    wait $pid1
    echo "First job (PID $pid1) completed."

    wait $pid2
    echo "Second job (PID $pid2) completed."

    wait $pid3
    echo "Third job (PID $pid3) completed."

fi

# convert to verl format
if [[ $GENERATE_VERL -ne 0 ]]; then
    for mode in rlvr sft
    do
        levels=(observational interventional counterfactual)
        if [[ "$mode" == "sft" ]]; then
            subfolder=sft
            data_prefix=data/sft
            extras=()
        elif [[ "$mode" == "rlvr" ]]; then
            subfolder=rlvr
            data_prefix=data/rl
            extras=()
        else
            echo "unknown mode ${mode}"
            exit 1
        fi
        echo "subfolder $mode, using data at $data_prefix"

        for level in ${levels[@]}
        do
            for graph_spec in n10v2
            do
                data_source=$data_prefix/${level}_distribution
                data_name=${level}_dist
                seed=42

                # training set 8k
                python3 -m data_generation.raw_to_verl \
                    --first 8000 \
                    --data_source $data_source \
                    --data_subsplits ${graph_spec} \
                    --local_dir data/verl/${subfolder}/${level}/${graph_spec}/8k/ \
                    --data_name ${data_name} \
                    --donot_name_by_difficulty \
                    --seed ${seed} \
                    "${extras[@]}"

            done
        done
    done
fi