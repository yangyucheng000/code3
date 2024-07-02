collection=didemo
visual_feature=resnet_slowfast
# root_path=data/
visual_feat_dim=4352

#CF
clip_scale_w=0.5 # 4
frame_scale_w=0.5 # 0.6
# eval
eval_context_bsz=500
eval_query_bsz=500

margin=0.2 #0.2
max_ctx_l=12 # 128

# proposals
width_lower_bound=0.1667
num_gauss_center=12 #6
num_gauss_width=6
num_props=6 # 8
map_size=6
npev=40

pooling_ksize=1
pooling_stride=1
conv_ksize=1

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
--clip_scale_w $clip_scale_w \
--frame_scale_w $frame_scale_w \
--device_ids $device_ids \
--visual_feat_dim $visual_feat_dim \
--eval_context_bsz $eval_context_bsz \
--eval_query_bsz $eval_query_bsz \
--max_ctx_l $max_ctx_l \
--margin $margin \
--num_props $num_props \
--width_lower_bound $width_lower_bound \
--num_gauss_center $num_gauss_center \
--num_gauss_width $num_gauss_width \
--map_size $map_size \
--npev $npev \
--pooling_ksize $pooling_ksize \
--pooling_stride $pooling_stride \
--conv_ksize $conv_ksize \
--global_sample # uncomment to use