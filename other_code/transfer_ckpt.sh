#!/bin/bash

# 路径前缀（注意这里不包括 ads-cli 命令）
src_dir="/mnt/afs/wyzhang/code/DreamVLA-seer/checkpoints/fine_tune_dreamvla_nodino_mlp_q9/nodino_mlp_q9"
dst_prefix="s3://7423E748AEE14F8B8ACDF13D47F6BCE9:557BB638422C442FA0779C0B7C0ACE41@checkpoint.aoss-internal.cn-sh-01b.sensecoreapi-oss.cn/wyzhang/nodino_mlp_q9_e"

# 上传编号范围
for i in {7..20}; do
    src_path="${src_dir}/${i}.pth"
    dst_path="${dst_prefix}${i}.pth"

    echo "Uploading ${src_path} -> ${dst_path}"
    ../../ads-cli cp "$src_path" "$dst_path"
done
