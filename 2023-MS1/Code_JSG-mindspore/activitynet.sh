collection=activitynet
visual_feature=i3d
# root_path=data/
# intra-margin
intra_margin=0.15 #0.2
# CF ratio
clip_scale_w=0.5
frame_scale_w=0.5
# eval
eval_context_bsz=500
eval_query_bsz=200 # 20000

# loss
alpha3=0 # for frame intra CL loss

bsz=256 # 128
lr=1.5e-4

# proposal
num_gauss_center=3 #32
num_gauss_width=4 #3, 10
num_props=4
width_lower_bound=0.25

# dynamic
exp_id=$1
device_ids=$2
root_path=$3

# training
python method/train.py  \
--collection $collection \
--visual_feature $visual_feature \
--root_path $root_path  \
--dset_name $collection \
--exp_id $exp_id \
--device_ids $device_ids \
--eval_query_bsz $eval_query_bsz \
--clip_scale_w $clip_scale_w \
--frame_scale_w $frame_scale_w \
--intra_margin $intra_margin \
--bsz $bsz \
--lr $lr \
--eval_context_bsz $eval_context_bsz \
--vocab_size $vocab_size \
--num_gauss_center $num_gauss_center \
--num_gauss_width $num_gauss_width \
--num_props $num_props \
--alpha3 $alpha3 \
--width_lower_bound $width_lower_bound \
--global_sample