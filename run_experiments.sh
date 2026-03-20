#!/bin/bash
# run_experiments.sh - 批量运行 CLAM 复现实验
# 用法: bash run_experiments.sh
#
# 特性:
#   - 自动跳过已完成的实验（通过检查结果 CSV）
#   - 每个实验结束后保存结果
#   - 支持断连恢复（checkpoint 机制）

CROPS=(0.08 0.2 0.3 0.4 0.5 0.6 0.8 1.0)
RESULTS_DIR="results"
CHECKPOINT_DIR="checkpoints"
mkdir -p $RESULTS_DIR
mkdir -p $CHECKPOINT_DIR

# 记录开始时间
echo "========================================"
echo "CLAM Reproduction Experiments"
echo "Started at: $(date)"
echo "========================================"

# 函数: 运行单个实验
run_exp() {
    local TASK=$1
    local METHOD=$2
    local METHOD_FLAG=$3
    local CROP=$4
    
    local EXP_ID="${TASK}_${METHOD}_crop${CROP}"
    local DONE_FILE="${RESULTS_DIR}/${EXP_ID}.done"
    
    # 跳过已完成的实验
    if [ -f "$DONE_FILE" ]; then
        echo "⏭️  ${EXP_ID} already done, skipping"
        return
    fi
    
    echo ""
    echo "========================================"
    echo "🚀 ${EXP_ID}"
    echo "   $(date)"
    echo "========================================"
    
    python train_CV.py \
        --task $TASK \
        $METHOD_FLAG \
        --crop_lower_bound $CROP \
        --num_workers 4 \
        --checkpoint_dir $CHECKPOINT_DIR
    
    # 保存结果
    if [ $? -eq 0 ]; then
        cp ${TASK}*.csv ${RESULTS_DIR}/ 2>/dev/null
        cp weights_${TASK}*.csv ${RESULTS_DIR}/ 2>/dev/null
        cp train_${TASK}*.csv ${RESULTS_DIR}/ 2>/dev/null
        touch "$DONE_FILE"
        echo "✅ ${EXP_ID} done!"
    else
        echo "❌ ${EXP_ID} failed!"
    fi
}

# ======== CIFAR-10: Normal + CLAM ========
echo ""
echo "===== CIFAR-10 ====="
for CROP in "${CROPS[@]}"; do
    run_exp "cifar10" "normal" "" "$CROP"
done
for CROP in "${CROPS[@]}"; do
    run_exp "cifar10" "CLAM" "--CLAM_loss true" "$CROP"
done

# ======== Fashion-MNIST: Normal + CLAM ========
echo ""
echo "===== Fashion-MNIST ====="
for CROP in "${CROPS[@]}"; do
    run_exp "fmnist" "normal" "" "$CROP"
done
for CROP in "${CROPS[@]}"; do
    run_exp "fmnist" "CLAM" "--CLAM_loss true" "$CROP"
done

# ======== CIFAR-100: Normal + CLAM ========
echo ""
echo "===== CIFAR-100 ====="
for CROP in "${CROPS[@]}"; do
    run_exp "cifar100" "normal" "" "$CROP"
done
for CROP in "${CROPS[@]}"; do
    run_exp "cifar100" "CLAM" "--CLAM_loss true" "$CROP"
done

echo ""
echo "========================================"
echo "All experiments finished at: $(date)"
echo "Results saved in: ${RESULTS_DIR}/"
echo "========================================"